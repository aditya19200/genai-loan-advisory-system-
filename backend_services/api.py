from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import UPLOADS_DIR, ensure_directories
from app.schemas import ChatMessage, ChatRequest, ChatResponse, DocumentUploadRequest, ExplanationResponse, FeedbackInput, PredictionInput, PredictionResponse
from backend_services.llm_service import chat_with_customer
from backend_services.pipeline import PredictionService
from database.sqlite_db import DatabaseManager
from rag.document_policy import extract_pdf_text
from rag.retriever import get_rbi_knowledge_base


ensure_directories()
service = PredictionService()
db = DatabaseManager()
rag = get_rbi_knowledge_base()
app = FastAPI(title="AI-Based Loan Decision System", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models/comparison")
def model_comparison():
    return service.model_comparison()


@app.get("/metrics/overview")
def metrics_overview():
    return {
        "model_comparison": service.model_comparison(),
        "operational": db.metrics_overview(),
    }


@app.get("/rag/status")
def rag_status():
    return rag.status()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput, background_tasks: BackgroundTasks) -> PredictionResponse:
    result = service.predict(payload)
    if result["explain_available"]:
        background_tasks.add_task(service.generate_explanation, result["request_id"])
    return PredictionResponse(
        request_id=result["request_id"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        decision=result["decision"],
        risk_score=result["risk_score"],
        explain_available=result["explain_available"],
        explanation_status=result["explanation_status"],
        created_at=datetime.fromisoformat(result["created_at"]),
    )


@app.get("/predictions")
def list_predictions():
    rows = db.fetch_all("predictions")
    return [
        {
            "request_id": row["request_id"],
            "decision": row.get("decision"),
            "risk_score": row.get("probability"),
            "created_at": row.get("created_at"),
        }
        for row in rows
    ]


@app.get("/predictions/{request_id}")
def get_prediction(request_id: str):
    row = db.fetch_one("predictions", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction request not found")
    return {
        "request_id": row["request_id"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "decision": row.get("decision"),
        "risk_score": row.get("probability"),
        "input_payload": json.loads(row["input_payload"]),
        "created_at": row["created_at"],
    }


@app.post("/documents/{request_id}/upload")
def upload_document(request_id: str, payload: DocumentUploadRequest):
    prediction_row = db.fetch_one("predictions", request_id)
    if not prediction_row:
        raise HTTPException(status_code=404, detail="Prediction request not found")
    if not payload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported for document-to-policy RAG")
    file_bytes = payload.decoded_bytes()
    extraction = extract_pdf_text(file_bytes)
    extracted_text = extraction.get("text", "")
    if not extracted_text:
        raise HTTPException(status_code=400, detail=extraction.get("details") or "The uploaded PDF did not contain extractable text")

    target_dir = Path(UPLOADS_DIR) / request_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(payload.filename).name
    target_path.write_bytes(file_bytes)
    db.upsert_uploaded_document(
        request_id,
        Path(payload.filename).name,
        str(target_path),
        extracted_text,
        extraction.get("source", "unavailable"),
        extraction.get("details", ""),
    )
    return {
        "status": "stored",
        "request_id": request_id,
        "filename": Path(payload.filename).name,
        "extracted_characters": len(extracted_text),
        "extraction_source": extraction.get("source", "unavailable"),
        "extraction_details": extraction.get("details", ""),
    }


@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: PredictionInput) -> ExplanationResponse:
    result = service.explain_input(payload)
    return ExplanationResponse(**result)


@app.post("/explanations/{request_id}/generate-sync", response_model=ExplanationResponse)
def generate_explanation_sync(request_id: str) -> ExplanationResponse:
    row = db.fetch_one("predictions", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction request not found")
    service.generate_explanation(request_id, True)
    explanation_row = db.fetch_one("explanations", request_id)
    if not explanation_row:
        raise HTTPException(status_code=404, detail="Explanation request not found")
    return ExplanationResponse(**service._compose_explanation_response(request_id, explanation_row))


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    message = payload.message
    applicant_context = payload.applicant_context
    use_rag = any(token in message.lower() for token in ["rbi", "guideline", "guidelines", "rule", "policy", "policies", "kfs", "ombudsman"])
    rag_context = rag.retrieve(f"User question: {message}. Applicant context: {applicant_context}") if use_rag or applicant_context else []
    rag_source = "retrieved" if rag_context else "unavailable"
    chat_result = chat_with_customer(
        message=message,
        history=[item.model_dump() for item in payload.history],
        applicant_context=applicant_context,
        rag_context=rag_context,
    )
    db.insert_chat_metric(
        source=chat_result["source"],
        rag_source=rag_source,
        latency_ms=(time.perf_counter() - started) * 1000,
        message_length=len(message),
    )
    updated_history = [*payload.history, ChatMessage(role="user", content=payload.message), ChatMessage(role="assistant", content=chat_result["reply"])]
    return ChatResponse(reply=chat_result["reply"], source=chat_result["source"], rag_source=rag_source, history=updated_history, rag_context=rag_context)


@app.get("/explanations/{request_id}", response_model=ExplanationResponse)
def get_explanation(request_id: str) -> ExplanationResponse:
    row = db.fetch_one("explanations", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Explanation request not found")
    return ExplanationResponse(**service._compose_explanation_response(request_id, row))


@app.post("/explanations/{request_id}/regenerate")
def regenerate_explanation(request_id: str, background_tasks: BackgroundTasks):
    row = db.fetch_one("predictions", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction request not found")
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE explanations
            SET status = ?, decision = NULL, risk_score = NULL, shap_global = NULL, shap_local = NULL,
                sentiment = NULL, explanation_text = NULL, advisory = NULL, counter_offer = NULL, reports = NULL,
                explanation_source = NULL, reports_source = NULL,
                rag_source = NULL,
                document_rag_source = NULL, document_extraction_source = NULL, document_report_source = NULL,
                llm_response = NULL, rag_context = NULL, document_rag_context = NULL,
                uploaded_document_name = NULL, generation_time_ms = NULL, generated_at = NULL
            WHERE request_id = ?
            """,
            ("pending", request_id),
        )
    background_tasks.add_task(service.generate_explanation, request_id, True)
    return {"status": "queued", "request_id": request_id}


@app.get("/fairness")
def fairness():
    return [
        {
            "model_name": record["name"],
            "model_version": record["version"],
            "fairness": record["metrics"]["fairness"],
        }
        for record in service.model_comparison()
    ]


@app.get("/monitoring")
def monitoring():
    return db.fetch_all("monitoring")


@app.get("/audit-logs")
def audit_logs():
    return db.fetch_all("audit_logs")


@app.post("/feedback")
def submit_feedback(payload: FeedbackInput):
    db.insert_feedback(payload.request_id, payload.rating, payload.comment)
    return {"status": "stored"}


@app.get("/feedback")
def get_feedback():
    return db.fetch_all("feedback")
