# API Documentation

## Endpoints

### `GET /health`
Returns service health.

### `GET /models/comparison`
Returns top registered models with versions and metrics.

### `POST /predict`
Returns a `request_id`, decision, risk score, chosen model, and explanation availability/status.

### `POST /explain`
Runs the explanation flow synchronously and returns SHAP output, sentiment, advisory, Gemini response data, and optional counter-offer text.

### `GET /explanations/{request_id}`
Returns status plus SHAP, explanation text, advisory, sentiment, and optional counter-offer when ready.

### `GET /fairness`
Returns fairness pre-checks for the top registered models.

### `GET /monitoring`
Returns latency and stability event history.

### `GET /audit-logs`
Returns the simulated immutable audit ledger.

### `POST /feedback`
Stores a user rating and comment for an explanation.
