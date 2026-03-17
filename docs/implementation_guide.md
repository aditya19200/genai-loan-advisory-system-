# Implementation Guide

## Modules

- `data/processing.py`: loads the dataset, cleans missing values, engineers features, identifies sensitive attributes, and prepares train/test splits.
- `models/training.py`: trains Logistic Regression, Random Forest, and XGBoost with grid search and cross-validation.
- `models/registry.py`: persists the top ranked models and versions in JSON.
- `models/evaluation.py`: computes standard classification metrics and fairness pre-checks.
- `explainability/engine.py`: generates SHAP, LIME, and ELI5 outputs and calculates explanation stability.
- `database/sqlite_db.py`: stores predictions, explanations, monitoring events, audit logs, and user feedback.
- `backend_services/api.py`: exposes the FastAPI interface and async explanation workflow.
- `frontend/streamlit_app.py`: renders the operational dashboard.
- `alibi_service/`: optional separate FastAPI service for counterfactual explanations in an isolated environment.

## Local workflow

1. Install dependencies from `requirements.txt`.
2. Start the API. The backend trains and registers models if artifacts are missing.
3. Start Streamlit and submit a prediction.
4. Refresh the explanation page after the background task completes.
