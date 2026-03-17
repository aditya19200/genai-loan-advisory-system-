# API Documentation

## Endpoints

### `GET /health`
Returns service health.

### `GET /models/comparison`
Returns top registered models with versions and metrics.

### `POST /predict`
Returns a `request_id`, prediction, probability, chosen model, and pending explanation status.

### `GET /explanations/{request_id}`
Returns status plus SHAP, LIME, ELI5, and stability metrics when ready.

### `GET /fairness`
Returns fairness pre-checks for the top registered models.

### `GET /monitoring`
Returns latency and stability event history.

### `GET /audit-logs`
Returns the simulated immutable audit ledger.

### `POST /counterfactuals/{request_id}/request`
Queues an optional Alibi-based counterfactual explanation request for the selected case.

### `GET /counterfactuals/{request_id}`
Returns the current status and result of the optional counterfactual explanation.

### `POST /feedback`
Stores a user rating and comment for an explanation.
