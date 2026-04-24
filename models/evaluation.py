from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fairness.metrics import compute_fairness_metrics


def _ks_statistic(y_true: pd.Series, y_proba: pd.Series) -> float:
    evaluation_frame = pd.DataFrame({"y_true": y_true, "y_proba": y_proba}).sort_values("y_proba")
    positives = max(int((evaluation_frame["y_true"] == 1).sum()), 1)
    negatives = max(int((evaluation_frame["y_true"] == 0).sum()), 1)
    evaluation_frame["cum_positive_rate"] = (evaluation_frame["y_true"] == 1).cumsum() / positives
    evaluation_frame["cum_negative_rate"] = (evaluation_frame["y_true"] == 0).cumsum() / negatives
    return float((evaluation_frame["cum_positive_rate"] - evaluation_frame["cum_negative_rate"]).abs().max())


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, sensitive_test: pd.DataFrame) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "ks_statistic": _ks_statistic(pd.Series(y_test), pd.Series(y_proba, index=y_test.index)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    metrics["fairness"] = compute_fairness_metrics(y_test, pd.Series(y_pred, index=y_test.index), sensitive_test)
    return metrics
