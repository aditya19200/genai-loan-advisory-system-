from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
except ImportError:  # pragma: no cover
    demographic_parity_difference = None
    equalized_odds_difference = None


def compute_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_features: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for column in sensitive_features.columns:
        feature = sensitive_features[column]
        if demographic_parity_difference is None or equalized_odds_difference is None:
            results[column] = {
                "demographic_parity_difference": None,
                "equal_opportunity_difference": None,
                "status": "fairlearn_not_installed",
            }
        else:
            results[column] = {
                "demographic_parity_difference": float(
                    demographic_parity_difference(y_true, y_pred, sensitive_features=feature)
                ),
                "equal_opportunity_difference": float(
                    equalized_odds_difference(y_true, y_pred, sensitive_features=feature)
                ),
            }
    return results
