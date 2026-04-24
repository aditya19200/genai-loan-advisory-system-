from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.schemas import PredictionInput
from audit.logger import AuditLogger
from backend_services.llm_service import (
    build_document_compliance_prompt,
    build_explanation_prompt,
    build_rule_based_response,
    call_gemini,
    ensure_rbi_report,
    should_generate_advisory,
    simple_sentiment,
)
from data.processing import DataProcessor
from database.sqlite_db import DatabaseManager
from explainability.engine import ExplainabilityEngine
from models.evaluation import evaluate_model
from models.registry import ModelRegistry
from models.training import ModelTrainer, load_registered_model
from monitoring.service import MonitoringService
from rag.document_policy import build_document_rag_query, fallback_document_report
from rag.retriever import build_rag_query, get_rbi_knowledge_base


class PredictionService:
    def __init__(self) -> None:
        self.db = DatabaseManager()
        self.monitoring = MonitoringService(self.db)
        self.audit = AuditLogger(self.db)
        self.registry = ModelRegistry()
        self.trainer = ModelTrainer()
        self.rag = get_rbi_knowledge_base()
        self._load_models()

    def _load_models(self) -> None:
        top_models = self.trainer.ensure_trained()
        self.top_models = top_models
        self.primary_record = top_models[0]
        try:
            self.primary_bundle = load_registered_model(self.primary_record)
            self.explainer = ExplainabilityEngine(self.primary_bundle)
        except Exception:
            refreshed = self.trainer.train_all()
            self.top_models = self.registry.top_models(limit=1) or [
                {
                    "name": refreshed[0].name,
                    "version": refreshed[0].version,
                    "artifact_path": refreshed[0].artifact_path,
                    "metrics": refreshed[0].metrics,
                    "rank": 1,
                }
            ]
            self.primary_record = self.top_models[0]
            self.primary_bundle = load_registered_model(self.primary_record)
            self.explainer = ExplainabilityEngine(self.primary_bundle)

    def model_comparison(self) -> list[dict[str, Any]]:
        prepared = self.trainer.processor.prepare_data()
        enriched_models: list[dict[str, Any]] = []
        for record in self.top_models:
            enriched = dict(record)
            try:
                bundle = load_registered_model(record)
                enriched["metrics"] = evaluate_model(
                    bundle["model"],
                    prepared.X_test,
                    prepared.y_test,
                    prepared.sensitive_test,
                )
            except Exception:
                enriched["metrics"] = record.get("metrics", {})
            enriched_models.append(enriched)
        return enriched_models

    def predict(self, payload: PredictionInput) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        input_payload = payload.model_dump()
        frame = DataProcessor.api_payload_to_frame(input_payload)
        model = self.primary_bundle["model"]
        risk_score = float(model.predict_proba(frame)[0][1])
        decision = "Rejected" if risk_score >= 0.5 else "Approved"
        created_at = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.perf_counter() - started) * 1000
        explanation_needed = should_generate_advisory(risk_score, payload.ask_explain)

        record = {
            "request_id": request_id,
            "model_name": self.primary_record["name"],
            "model_version": self.primary_record["version"],
            "input_payload": input_payload,
            "prediction": int(risk_score >= 0.5),
            "decision": decision,
            "probability": risk_score,
            "created_at": created_at,
        }
        self.db.insert_prediction(record, explanation_status="pending" if explanation_needed else "not_requested")
        self.monitoring.record_prediction(request_id, latency_ms)
        return {
            **record,
            "risk_score": risk_score,
            "explain_available": explanation_needed,
            "explanation_status": "pending" if explanation_needed else "not_requested",
        }

    def explain_input(self, payload: PredictionInput) -> dict[str, Any]:
        prediction = self.predict(payload)
        if not prediction["explain_available"]:
            self.generate_explanation(prediction["request_id"], force=True)
        else:
            self.generate_explanation(prediction["request_id"])
        explanation_row = self.db.fetch_one("explanations", prediction["request_id"]) or {}
        return self._compose_explanation_response(prediction["request_id"], explanation_row)

    def generate_explanation(self, request_id: str, force: bool = False) -> None:
        prediction_row = self.db.fetch_one("predictions", request_id)
        if not prediction_row:
            return
        input_payload = json.loads(prediction_row["input_payload"])
        risk_score = float(prediction_row["probability"])
        should_run = should_generate_advisory(risk_score, bool(input_payload.get("ask_explain")))
        if not should_run and not force:
            return

        started = time.perf_counter()
        try:
            frame = DataProcessor.api_payload_to_frame(input_payload)
            explanation = self.explainer.explain(frame)
            sentiment = simple_sentiment(input_payload.get("user_text"))
            rag_context = self.rag.retrieve(
                build_rag_query(
                    prediction_row["decision"],
                    risk_score,
                    explanation["shap_local"],
                    input_payload.get("user_text"),
                )
            )
            rag_source = "retrieved" if rag_context else "unavailable"
            fallback = build_rule_based_response(
                decision=prediction_row["decision"],
                risk_score=risk_score,
                shap_explanation=explanation["shap_local"],
                sentiment=sentiment,
                sample=input_payload,
                rag_context=rag_context,
            )
            llm_response: dict[str, Any] = {}
            if should_run:
                try:
                    llm_response = call_gemini(
                        build_explanation_prompt(
                            prediction_row["decision"],
                            risk_score,
                            explanation["shap_local"],
                            sentiment,
                            rag_context,
                        )
                    )
                except Exception as exc:
                    llm_response = {"error": str(exc), "parsed": {}}

            parsed = llm_response.get("parsed", {}) if isinstance(llm_response, dict) else {}
            explanation_source = "gemini" if parsed.get("explanation") else "fallback"
            reports = ensure_rbi_report(parsed.get("reports") or fallback["reports"], fallback["reports"])
            reports_source = "gemini" if parsed.get("reports") else "fallback"
            uploaded_document = self.db.fetch_uploaded_document(request_id)
            document_rag_context: list[dict[str, Any]] = []
            document_rag_source = "unavailable"
            document_extraction_source = "unavailable"
            document_report_source = "fallback"
            uploaded_document_name = None
            if uploaded_document:
                uploaded_document_name = uploaded_document.get("filename")
                document_extraction_source = uploaded_document.get("extraction_source") or "unavailable"
                document_rag_context = self.rag.retrieve(
                    build_document_rag_query(uploaded_document.get("extracted_text", ""), input_payload.get("user_text"))
                )
                document_rag_source = "retrieved" if document_rag_context else "unavailable"
                document_report = fallback_document_report(
                    uploaded_document_name or "uploaded.pdf",
                    uploaded_document.get("extracted_text", ""),
                    document_rag_context,
                )
                try:
                    compliance_response = call_gemini(
                        build_document_compliance_prompt(
                            uploaded_document_name or "uploaded.pdf",
                            uploaded_document.get("extracted_text", ""),
                            document_rag_context,
                        ),
                        response_schema={
                            "type": "OBJECT",
                            "properties": {
                                "summary": {"type": "STRING"},
                                "satisfied": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "missing": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "unclear": {"type": "ARRAY", "items": {"type": "STRING"}},
                                "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
                            },
                            "required": ["summary", "satisfied", "missing", "unclear", "evidence"],
                        },
                    )
                    compliance_parsed = compliance_response.get("parsed", {})
                    if compliance_parsed.get("summary"):
                        bullets = (
                            [*(f"Satisfied: {item}" for item in compliance_parsed.get("satisfied", []))]
                            + [*(f"Missing: {item}" for item in compliance_parsed.get("missing", []))]
                            + [*(f"Unclear: {item}" for item in compliance_parsed.get("unclear", []))]
                            + [*(f"Evidence: {item}" for item in compliance_parsed.get("evidence", [])[:4])]
                        )
                        document_report = {
                            "title": "Uploaded Document Compliance Report",
                            "audience": "User and Loan Officer",
                            "summary": compliance_parsed["summary"],
                            "bullets": bullets,
                        }
                        document_report_source = "gemini"
                except Exception:
                    pass
                reports = [*reports, document_report]

            duration_ms = (time.perf_counter() - started) * 1000
            payload = {
                "decision": prediction_row["decision"],
                "risk_score": risk_score,
                "shap_global": explanation["shap_global"],
                "shap_local": explanation["shap_local"],
                "sentiment": sentiment,
                "explanation_text": parsed.get("explanation") or fallback["explanation_text"],
                "advisory": parsed.get("advisory") or fallback["advisory"],
                "counter_offer": parsed.get("counter_offer") or fallback["counter_offer"],
                "reports": reports,
                "explanation_source": explanation_source,
                "reports_source": reports_source,
                "rag_source": rag_source,
                "document_rag_source": document_rag_source,
                "document_extraction_source": document_extraction_source,
                "document_report_source": document_report_source,
                "llm_response": llm_response,
                "rag_context": rag_context,
                "document_rag_context": document_rag_context,
                "uploaded_document_name": uploaded_document_name,
            }
            self.db.insert_explanation(request_id, payload, duration_ms)
            self.monitoring.record_explanation(request_id, duration_ms, None)
            audit_payload = {
                "input_payload": input_payload,
                "decision": prediction_row["decision"],
                "risk_score": risk_score,
                "explanation": payload,
                "model_version": prediction_row["model_version"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.audit.log(request_id, prediction_row["model_version"], audit_payload)
        except Exception as exc:
            self.db.mark_explanation_error(request_id, f"Explanation generation failed: {exc}")

    def _compose_explanation_response(self, request_id: str, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "status": row.get("status", "not_requested"),
            "decision": row.get("decision"),
            "risk_score": row.get("risk_score"),
            "shap_global": json.loads(row.get("shap_global") or "[]"),
            "shap_local": json.loads(row.get("shap_local") or "[]"),
            "sentiment": row.get("sentiment") or "neutral",
            "explanation_text": row.get("explanation_text") or "",
            "advisory": row.get("advisory") or "",
            "counter_offer": row.get("counter_offer"),
            "reports": json.loads(row.get("reports") or "[]"),
            "explanation_source": row.get("explanation_source") or "fallback",
            "reports_source": row.get("reports_source") or "fallback",
            "rag_source": row.get("rag_source") or "unavailable",
            "document_rag_source": row.get("document_rag_source") or "unavailable",
            "document_extraction_source": row.get("document_extraction_source") or "unavailable",
            "document_report_source": row.get("document_report_source") or "fallback",
            "llm_response": json.loads(row.get("llm_response") or "{}"),
            "rag_context": json.loads(row.get("rag_context") or "[]"),
            "document_rag_context": json.loads(row.get("document_rag_context") or "[]"),
            "uploaded_document_name": row.get("uploaded_document_name"),
            "generated_at": datetime.fromisoformat(row["generated_at"]) if row.get("generated_at") else None,
        }
