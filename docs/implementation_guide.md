# Implementation Guide

## Modules

- `data/processing.py`: loads the German credit dataset, derives the HLD-facing loan features, identifies sensitive attributes, and prepares train/test splits.
- `models/training.py`: trains the primary XGBoost credit risk model and persists the registered artifact.
- `models/registry.py`: persists the top ranked model and version in JSON.
- `models/evaluation.py`: computes standard classification metrics and fairness pre-checks.
- `explainability/engine.py`: generates SHAP global and local importance outputs.
- `database/sqlite_db.py`: stores predictions, explanations, monitoring events, audit logs, and user feedback.
- `backend_services/api.py`: exposes the FastAPI decision, async explanation, and synchronous explanation workflow.
- `backend_services/llm_service.py`: builds Gemini prompts, fallback advisory text, and sentiment handling.
- `frontend/streamlit_app.py`: renders the operational dashboard.

## Local workflow

1. Install dependencies from `requirements.txt`.
2. Set `GEMINI_API_KEY` if live LLM responses are required.
3. Start the API. The backend trains and registers the current model if artifacts are missing or incompatible.
4. Start Streamlit and submit a prediction.
5. Refresh the explanation page after the background task completes, or use the synchronous explain action.
