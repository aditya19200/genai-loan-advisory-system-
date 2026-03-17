from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from fairness.metrics import compute_fairness_metrics


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, sensitive_test: pd.DataFrame) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    metrics["fairness"] = compute_fairness_metrics(y_test, pd.Series(y_pred, index=y_test.index), sensitive_test)
    return metrics
