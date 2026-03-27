from __future__ import annotations

from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ensure_directories
from app.schemas import ChatMessage, ChatRequest, ChatResponse, ExplanationResponse, FeedbackInput, PredictionInput, PredictionResponse
from backend_services.llm_service import chat_with_customer
from backend_services.pipeline import PredictionService
from database.sqlite_db import DatabaseManager


ensure_directories()
service = PredictionService()
db = DatabaseManager()
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
        "input_payload": __import__("json").loads(row["input_payload"]),
        "created_at": row["created_at"],
    }


@app.post("/explain", response_model=ExplanationResponse)
def explain(payload: PredictionInput) -> ExplanationResponse:
    result = service.explain_input(payload)
    return ExplanationResponse(**result)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    chat_result = chat_with_customer(
        message=payload.message,
        history=[item.model_dump() for item in payload.history],
        applicant_context=payload.applicant_context,
    )
    updated_history = [*payload.history, ChatMessage(role="user", content=payload.message), ChatMessage(role="assistant", content=chat_result["reply"])]
    return ChatResponse(reply=chat_result["reply"], source=chat_result["source"], history=updated_history)


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
                llm_response = NULL, rag_context = NULL, generation_time_ms = NULL, generated_at = NULL
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
