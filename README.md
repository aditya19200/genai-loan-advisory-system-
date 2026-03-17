# XAI Finance Platform

GitHub-ready prototype of an explainable AI financial decision platform built with FastAPI, Streamlit, scikit-learn, XGBoost, SHAP, LIME, ELI5, Fairlearn, and SQLite.

## Python version

Use Python `3.11` or `3.12`. The scientific stack in this project is not reliable on the current Windows `3.13` setup and can fail during `numpy` import before the backend starts.

## What it demonstrates

- modular data processing and feature engineering
- model training with hyperparameter tuning and cross-validation
- model versioning and registry
- evaluation with accuracy, precision, recall, F1, ROC-AUC, and fairness pre-checks
- asynchronous explanation generation
- SHAP, LIME, and ELI5 explanation workflows
- SQLite-backed audit logging, monitoring, and feedback capture
- Streamlit dashboard for model, fairness, monitoring, and audit views

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

## Optional Alibi Counterfactual Service

Counterfactual explanations are isolated in a separate local service so the main app stays stable.

1. Create a separate environment for Alibi:

```bash
py -3.10 -m venv venv_alibi
venv_alibi\Scripts\activate
pip install -r requirements-alibi.txt
```

2. Start the Alibi service:

```bash
uvicorn alibi_service.api:app --reload --app-dir .
```

3. In the Streamlit explainability page, use `Generate counterfactual` for a request ID.

On Windows PowerShell, you can also use `.\run_app.ps1`.

## Render deployment prep

This repo includes a Render blueprint in `render.yaml` for:

- a Streamlit frontend web service
- a private FastAPI backend service
- a private optional Alibi counterfactual service

Important deployment notes:

- the frontend and backend must use environment-based service URLs, not `localhost`
- SQLite should be mounted on a persistent disk for the backend
- the Alibi service should stay isolated from the main app runtime
- for production-grade deployment, moving from SQLite to Postgres would be a stronger long-term choice
