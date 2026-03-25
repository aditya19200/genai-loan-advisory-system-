# AI-Based Loan Decision System

Integrated loan decision prototype built with FastAPI, Streamlit, scikit-learn, XGBoost, SHAP, Gemini, Fairlearn, and SQLite.

## Python version

Use Python `3.11` or `3.12`. The scientific stack in this project is not reliable on the current Windows `3.13` setup and can fail during `numpy` import before the backend starts.

## What it demonstrates

- modular data processing and feature engineering
- XGBoost-based credit risk training with feature engineering
- model versioning and registry
- evaluation with accuracy, precision, recall, F1, ROC-AUC, and fairness pre-checks
- immediate decision plus asynchronous or on-demand explanation generation
- SHAP-only explanation workflow aligned with the project HLD
- Gemini-backed explanation, advisory, and counter-offer generation
- SQLite-backed audit logging, monitoring, and feedback capture
- Streamlit dashboard for loan assessment, explainability, fairness, monitoring, and audit views

## Important prototype note

The provided German credit CSV does not contain a labeled target. This prototype derives a transparent heuristic `credit_risk` target so the platform can run end to end with the supplied dataset. Replace this with a real labeled target before any serious use.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend_services.api:app --reload --app-dir .
streamlit run frontend/streamlit_app.py
```

## Gemini configuration

Set `GEMINI_API_KEY` before starting the backend if you want live LLM-generated explanations. If it is not set, the backend falls back to deterministic rule-based explanation text and advisory.

On Windows PowerShell, you can also use `.\run_app.ps1`.

## Render deployment prep

This repo includes a Render blueprint in `render.yaml` for:

- a Streamlit frontend web service
- a private FastAPI backend service

Important deployment notes:

- the frontend and backend must use environment-based service URLs, not `localhost`
- SQLite should be mounted on a persistent disk for the backend
- set `GEMINI_API_KEY` in the backend service environment
- for production-grade deployment, moving from SQLite to Postgres would be a stronger long-term choice
