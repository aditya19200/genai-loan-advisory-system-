"""Data processing helpers.

Contains a small helper to prepare feature vectors for training/inference and a
feature schema for the loan model. Keep the feature order consistent across
training and serving.
"""
from __future__ import annotations

from typing import Any, Dict, List

FEATURE_ORDER: List[str] = [
    "income",
    "credit_score",
    "dti",
    "employment_length",
    "existing_loans",
    # add other features as needed
]

ADVISORY_FIELDS: List[str] = [
    "loan_amount",
    "tenure_months",
]


def normalize_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map user-provided payload to the canonical feature set. Fill missing keys with None.

    This function keeps the contract of input features stable.
    """
    out = {k: payload.get(k, None) for k in FEATURE_ORDER + ADVISORY_FIELDS}

    # Basic casting
    if out.get("credit_score") is not None:
        try:
            out["credit_score"] = float(out["credit_score"])
        except Exception:
            out["credit_score"] = None

    if out.get("income") is not None:
        try:
            out["income"] = float(out["income"])
        except Exception:
            out["income"] = None

    if out.get("dti") is not None:
        try:
            out["dti"] = float(out["dti"])
        except Exception:
            out["dti"] = None

    if out.get("employment_length") is not None:
        try:
            out["employment_length"] = float(out["employment_length"])
        except Exception:
            out["employment_length"] = None

    if out.get("existing_loans") is not None:
        try:
            out["existing_loans"] = float(out["existing_loans"])
        except Exception:
            out["existing_loans"] = None

    if out.get("loan_amount") is not None:
        try:
            out["loan_amount"] = float(out["loan_amount"])
        except Exception:
            out["loan_amount"] = None

    if out.get("tenure_months") is not None:
        try:
            out["tenure_months"] = int(float(out["tenure_months"]))
        except Exception:
            out["tenure_months"] = None

    return out
