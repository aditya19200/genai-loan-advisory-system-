from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.schemas import PredictionInput
from audit.logger import AuditLogger
from backend_services.counterfactual_client import CounterfactualClient
from data.processing import DataProcessor
from database.sqlite_db import DatabaseManager
from explainability.engine import ExplainabilityEngine
from models.registry import ModelRegistry
from models.training import ModelTrainer, load_registered_model
from monitoring.service import MonitoringService


class PredictionService:
    def __init__(self) -> None:
        self.db = DatabaseManager()
        self.monitoring = MonitoringService(self.db)
        self.audit = AuditLogger(self.db)
        self.counterfactual_client = CounterfactualClient()
        self.registry = ModelRegistry()
        self.trainer = ModelTrainer()
        self._load_models()

    def _load_models(self) -> None:
        top_models = self.trainer.ensure_trained()
        self.top_models = top_models
        self.primary_record = top_models[0]
        self.primary_bundle = load_registered_model(self.primary_record)
        self.explainer = ExplainabilityEngine(self.primary_bundle)

    def model_comparison(self) -> list[dict[str, Any]]:
        return self.top_models

    def predict(self, payload: PredictionInput) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        input_payload = payload.model_dump()
        frame = DataProcessor.api_payload_to_frame(input_payload)
        model = self.primary_bundle["model"]
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][1])
        created_at = datetime.now(timezone.utc).isoformat()
        latency_ms = (time.perf_counter() - started) * 1000
        record = {
            "request_id": request_id,
            "model_name": self.primary_record["name"],
            "model_version": self.primary_record["version"],
            "input_payload": input_payload,
            "prediction": prediction,
            "probability": probability,
            "created_at": created_at,
        }
        self.db.insert_prediction(record)
        self.monitoring.record_prediction(request_id, latency_ms)
        return record

    def generate_explanation(self, request_id: str) -> None:
        prediction_row = self.db.fetch_one("predictions", request_id)
        if not prediction_row:
            return
        started = time.perf_counter()
        try:
            input_payload = json.loads(prediction_row["input_payload"])
            frame = DataProcessor.api_payload_to_frame(input_payload)
            explanation = self.explainer.explain(frame)
            explanation["stability_score"] = self.explainer.stability_score(
                explanation["shap_local"], explanation["lime_local"]
            )
            duration_ms = (time.perf_counter() - started) * 1000
            self.db.insert_explanation(request_id, explanation, duration_ms)
            self.monitoring.record_explanation(request_id, duration_ms, explanation["stability_score"])
            audit_payload = {
                "prediction": prediction_row["prediction"],
                "probability": prediction_row["probability"],
                "explanation": explanation,
                "model_version": prediction_row["model_version"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.audit.log(request_id, prediction_row["model_version"], audit_payload)
        except Exception as exc:
            self.db.mark_explanation_error(request_id, f"Explanation generation failed: {exc}")

    def request_counterfactual(self, request_id: str) -> None:
        prediction_row = self.db.fetch_one("predictions", request_id)
        if not prediction_row:
            self.db.upsert_counterfactual(request_id, status="error", error_message="Prediction request not found.")
            return
        try:
            self.db.upsert_counterfactual(request_id, status="processing")
            payload = {
                "request_id": request_id,
                "input_payload": json.loads(prediction_row["input_payload"]),
                "model_artifact_path": self.primary_record["artifact_path"],
            }
            result, duration_ms = self.counterfactual_client.generate(payload)
            self.db.upsert_counterfactual(request_id, status="ready", result=result, generation_time_ms=duration_ms)
        except Exception as exc:
            self.db.upsert_counterfactual(request_id, status="error", error_message=f"Counterfactual generation failed: {exc}")
