from __future__ import annotations

import json
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ensure_directories
from app.schemas import ExplanationResponse, FeedbackInput, PredictionInput, PredictionResponse
from backend_services.pipeline import PredictionService
from database.sqlite_db import DatabaseManager


ensure_directories()
service = PredictionService()
db = DatabaseManager()
app = FastAPI(title="XAI Finance Platform", version="0.1.0")

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
    background_tasks.add_task(service.generate_explanation, result["request_id"])
    return PredictionResponse(
        request_id=result["request_id"],
        model_name=result["model_name"],
        model_version=result["model_version"],
        prediction=result["prediction"],
        probability=result["probability"],
        explanation_status="pending",
        created_at=datetime.fromisoformat(result["created_at"]),
    )


@app.get("/explanations/{request_id}", response_model=ExplanationResponse)
def get_explanation(request_id: str) -> ExplanationResponse:
    row = db.fetch_one("explanations", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Explanation request not found")
    return ExplanationResponse(
        request_id=request_id,
        status=row["status"],
        shap_global=json.loads(row["shap_global"] or "[]"),
        shap_local=json.loads(row["shap_local"] or "[]"),
        lime_local=json.loads(row["lime_local"] or "[]"),
        rule_based_explanation=json.loads(row["rule_based_explanation"] or "{}"),
        eli5_summary=row["eli5_summary"] or "",
        stability_score=row["stability_score"],
        generated_at=datetime.fromisoformat(row["generated_at"]) if row["generated_at"] else None,
    )


@app.post("/explanations/{request_id}/regenerate")
def regenerate_explanation(request_id: str, background_tasks: BackgroundTasks):
    row = db.fetch_one("predictions", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction request not found")
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE explanations
            SET status = ?, shap_global = NULL, shap_local = NULL, lime_local = NULL,
                rule_based_explanation = NULL, eli5_summary = NULL, stability_score = NULL,
                generation_time_ms = NULL, generated_at = NULL
            WHERE request_id = ?
            """,
            ("pending", request_id),
        )
    background_tasks.add_task(service.generate_explanation, request_id)
    return {"status": "queued", "request_id": request_id}


@app.post("/counterfactuals/{request_id}/request")
def request_counterfactual(request_id: str, background_tasks: BackgroundTasks):
    row = db.fetch_one("predictions", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Prediction request not found")
    db.upsert_counterfactual(request_id, status="queued")
    background_tasks.add_task(service.request_counterfactual, request_id)
    return {"status": "queued", "request_id": request_id}


@app.get("/counterfactuals/{request_id}")
def get_counterfactual(request_id: str):
    row = db.fetch_one("counterfactuals", request_id)
    if not row:
        raise HTTPException(status_code=404, detail="Counterfactual request not found")
    return {
        "request_id": request_id,
        "status": row["status"],
        "result": json.loads(row["result"] or "{}"),
        "error_message": row["error_message"],
        "generation_time_ms": row["generation_time_ms"],
        "generated_at": datetime.fromisoformat(row["generated_at"]) if row["generated_at"] else None,
    }


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
