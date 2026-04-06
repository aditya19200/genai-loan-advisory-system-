"""FastAPI application wiring the ML decision layer and conditional generative layer.

Routes:
- POST /predict -> returns immediate decision + risk
- POST /explain -> runs conditional explainability (SHAP), RAG, sentiment, and LLM (only when requested or borderline)
- POST /train -> convenience endpoint to trigger training (protected in prod)
"""
from __future__ import annotations

from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services import llm_service, ml_services, rag_service
from app.utils.processor import FEATURE_ORDER, normalize_input
from app.routers import upload as upload_router

app = FastAPI(title="Explainable Loan Decision API")
app.include_router(upload_router.router)


class PredictRequest(BaseModel):
    income: float
    credit_score: float
    dti: float
    employment_length: float
    existing_loans: float
    loan_amount: float | None = None
    tenure_months: int | None = None
    # optional flag to request explanation
    ask_explain: bool = False
    user_text: str | None = None


class TrainRequest(BaseModel):
    data_csv: str = "app/Data/processed_loan_data.csv"
    target_col: str = "Risk"


@app.on_event("startup")
def startup_load():
    # lazy: do not load heavy artifacts until used. But load preprocessor & model if present.
    try:
        app.state.model = ml_services.load_model()
        app.state.preprocessor = ml_services.load_preprocessor()
    except Exception:
        app.state.model = None
        app.state.preprocessor = None


@app.get("/")
def root():
    return {
        "message": "Explainable Loan Decision API is running.",
        "note": "Use POST /predict for the immediate loan decision and POST /explain for SHAP-based explanations.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "explain": "/explain",
        "train": "/train",
        "example_predict_request": {
            "income": 90000,
            "credit_score": 780,
            "dti": 0.18,
            "employment_length": 10,
            "existing_loans": 0,
            "loan_amount": 500000,
            "tenure_months": 36,
            "ask_explain": False,
        },
        "example_predict_response": {
            "decision": "Approved",
            "risk_score": 0.12,
            "explain_available": False,
        },
        "example_explain_request": {
            "income": 25000,
            "credit_score": 580,
            "dti": 0.62,
            "employment_length": 1,
            "existing_loans": 3,
            "loan_amount": 300000,
            "tenure_months": 24,
            "ask_explain": True,
            "user_text": "Why was my loan rejected?",
        },
        "example_explain_response": {
            "decision": "Rejected",
            "risk_score": 0.95,
            "shap": {
                "income": -0.2,
                "credit_score": 0.4,
                "dti": 0.5,
                "employment_length": 0.1,
                "existing_loans": 0.2,
            },
            "sentiment": "neutral",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "preprocessor_loaded": app.state.preprocessor is not None,
    }


@app.get("/demo")
def demo():
    if app.state.model is None or app.state.preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not available. Train or load a model first.")

    approved_sample = normalize_input(
        {
            "income": 90000,
            "credit_score": 780,
            "dti": 0.18,
            "employment_length": 10,
            "existing_loans": 0,
            "loan_amount": 500000,
            "tenure_months": 36,
        }
    )
    rejected_sample = normalize_input(
        {
            "income": 25000,
            "credit_score": 580,
            "dti": 0.62,
            "employment_length": 1,
            "existing_loans": 3,
            "loan_amount": 300000,
            "tenure_months": 24,
        }
    )
    borderline_sample = normalize_input(
        {
            "income": 35000,
            "credit_score": 660,
            "dti": 0.35,
            "employment_length": 6,
            "existing_loans": 1,
            "loan_amount": 400000,
            "tenure_months": 36,
        }
    )

    approved_decision, approved_risk = ml_services.predict_single(
        app.state.model, app.state.preprocessor, approved_sample, FEATURE_ORDER
    )
    rejected_decision, rejected_risk = ml_services.predict_single(
        app.state.model, app.state.preprocessor, rejected_sample, FEATURE_ORDER
    )
    borderline_decision, borderline_risk = ml_services.predict_single(
        app.state.model, app.state.preprocessor, borderline_sample, FEATURE_ORDER
    )

    rejected_shap = ml_services.explain_shap(
        app.state.model,
        app.state.preprocessor,
        rejected_sample,
        FEATURE_ORDER,
        background_df=_get_dummy_background_df(),
    )
    borderline_shap = ml_services.explain_shap(
        app.state.model,
        app.state.preprocessor,
        borderline_sample,
        FEATURE_ORDER,
        background_df=_get_dummy_background_df(),
    )
    borderline_advisory = llm_service.build_rule_based_response(
        borderline_decision, borderline_risk, borderline_shap, "neutral", sample=borderline_sample
    )

    return {
        "message": "Live model-generated demo output.",
        "approved_case": {
            "input": approved_sample,
            "decision": approved_decision,
            "risk_score": approved_risk,
        },
        "rejected_case": {
            "input": rejected_sample,
            "decision": rejected_decision,
            "risk_score": rejected_risk,
            "shap": rejected_shap,
        },
        "borderline_case": {
            "input": borderline_sample,
            "decision": borderline_decision,
            "risk_score": borderline_risk,
            "shap": borderline_shap,
            "counter_offer": borderline_advisory["counter_offer"],
            "advisory": borderline_advisory["advisory"],
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    payload = req.model_dump()
    normalized = normalize_input(payload)

    # ensure model loaded
    if app.state.model is None or app.state.preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not available. Train or load a model first.")

    decision, risk = ml_services.predict_single(app.state.model, app.state.preprocessor, normalized, FEATURE_ORDER)

    result = {"decision": decision, "risk_score": risk}

    # if ask_explain OR borderline risk, run explanation flow
    borderline = 0.4 <= risk <= 0.6
    if req.ask_explain or borderline:
        # We return a short placeholder; for full explanation call /explain
        result["explain_available"] = True
        result["message"] = "Explanation available via /explain endpoint (on-demand)."
    else:
        result["explain_available"] = False

    return result


@app.post("/explain")
def explain(req: PredictRequest):
    payload = req.model_dump()
    normalized = normalize_input(payload)

    if app.state.model is None or app.state.preprocessor is None:
        raise HTTPException(status_code=500, detail="Model not available. Train or load a model first.")

    decision, risk = ml_services.predict_single(app.state.model, app.state.preprocessor, normalized, FEATURE_ORDER)

    # Determine whether to run generative layer
    run_gen = req.ask_explain or (0.4 <= risk <= 0.6)

    response: Dict[str, Any] = {"decision": decision, "risk_score": risk}

    # Always produce SHAP explanation when explanation requested or borderline
    shap_expl = {}
    if run_gen:
        try:
            shap_expl = ml_services.explain_shap(
                app.state.model,
                app.state.preprocessor,
                normalized,
                FEATURE_ORDER,
                background_df=_get_dummy_background_df(),
            )
        except Exception as exc:
            response["shap_error"] = str(exc)
    response["shap"] = shap_expl

    # RAG: query regulatory context based on simple question
    rag_hits = []
    try:
        if run_gen:
            # Query using a short prompt about decision
            q = f"Loan decision justification for features: {normalized}"
            rag_hits = rag_service.query_similar(q, top_k=3)
            response["rag"] = rag_hits
    except Exception as e:
        response["rag_error"] = str(e)

    # sentiment (simple heuristic using user_text presence)
    sentiment = _simple_sentiment(req.user_text) if req.user_text else "neutral"
    response["sentiment"] = sentiment
    fallback_response = llm_service.build_rule_based_response(
        decision, risk, shap_expl or {}, sentiment, sample=normalized
    )
    response["explanation_text"] = fallback_response["explanation"]
    response["advisory"] = fallback_response["advisory"]
    response["counter_offer"] = fallback_response["counter_offer"]

    # Call LLM only if run_gen
    if run_gen:
        try:
            prompt = llm_service.build_explanation_prompt(decision, risk, shap_expl or {}, rag_hits or [], sentiment)
            llm_out = llm_service.call_llm(prompt)
            response["llm"] = llm_out
        except Exception as exc:
            response["llm_error"] = str(exc)

    return response


def _simple_sentiment(text: str | None) -> str:
    if not text:
        return "neutral"
    txt = text.lower()
    if any(w in txt for w in ["angry", "frustrat", "upset", "annoyed"]):
        return "frustrated"
    if any(w in txt for w in ["thanks", "thank", "great", "good", "happy"]):
        return "positive"
    return "neutral"


def _get_dummy_background_df():
    """Produces a small DataFrame used for SHAP background. In production, use a representative sample from training data.

    Here we synthesize zeros — callers should replace with real background.
    """
    import pandas as pd

    data = {k: [0.0] for k in FEATURE_ORDER}
    return pd.DataFrame(data)


@app.post("/train")
def train(req: TrainRequest):
    # WARNING: this endpoint should be protected in production (not exposed publically)
    # It expects a path to a CSV accessible to the service.
    feature_cols = FEATURE_ORDER
    out = ml_services.train_xgboost(req.data_csv, feature_cols, target_col=req.target_col)
    # reload model into app state
    try:
        app.state.model = ml_services.load_model()
        app.state.preprocessor = ml_services.load_preprocessor()
    except Exception:
        pass
    return out
