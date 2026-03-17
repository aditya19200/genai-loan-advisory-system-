from __future__ import annotations

from typing import Any

import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference


def compute_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for column in sensitive_features.columns:
        feature = sensitive_features[column]
        results[column] = {
            "demographic_parity_difference": float(
                demographic_parity_difference(y_true, y_pred, sensitive_features=feature)
            ),
            "equal_opportunity_difference": float(
                equalized_odds_difference(y_true, y_pred, sensitive_features=feature)
            ),
        }
    return results
