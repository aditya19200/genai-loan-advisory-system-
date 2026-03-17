# Architecture

This prototype implements a local Explainable AI financial decision platform with a modular Python stack.

## Assumption

The provided `data/german_credit.csv` file does not include a ground-truth default label. To keep the prototype runnable, the platform derives a transparent binary `credit_risk` target using repayment burden, duration, account liquidity, housing, and purpose risk heuristics. This is for demonstration only.

## System architecture

```mermaid
flowchart LR
    UI[Streamlit Dashboard] --> API[FastAPI Backend]
    API --> DP[Data Processing]
    API --> MR[Model Registry]
    API --> DB[(SQLite)]
    API --> BG[Background Explanation Tasks]
    BG --> EX[SHAP / LIME / ELI5]
    API --> ALIBI[Optional Alibi Service]
    ALIBI --> API
    BG --> AUD[Audit Hash Logging]
    BG --> MON[Monitoring]
    MR --> M1[Logistic Regression]
    MR --> M2[Random Forest]
    MR --> M3[XGBoost]
```

## Component diagram

```mermaid
flowchart TD
    A[data/processing.py]
    B[models/training.py]
    C[models/evaluation.py]
    D[explainability/engine.py]
    E[fairness/metrics.py]
    F[audit/logger.py]
    G[monitoring/service.py]
    H[backend_services/api.py]
    I[frontend/streamlit_app.py]
    J[database/sqlite_db.py]

    H --> A
    H --> B
    H --> D
    H --> F
    H --> G
    H --> J
    I --> H
    B --> C
    C --> E
```

## Data flow diagram

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant FastAPI
    participant Models
    participant Background
    participant SQLite

    User->>Streamlit: Enter applicant profile
    Streamlit->>FastAPI: POST /predict
    FastAPI->>Models: Predict synchronously
    Models-->>FastAPI: prediction + probability
    FastAPI->>SQLite: Store prediction
    FastAPI-->>Streamlit: request_id + prediction
    FastAPI->>Background: schedule explanation
    Background->>Models: SHAP / LIME / ELI5
    Background->>SQLite: store explanation + metrics + audit hash
    Streamlit->>FastAPI: GET /explanations/{request_id}
    FastAPI-->>Streamlit: explanation payload
```
