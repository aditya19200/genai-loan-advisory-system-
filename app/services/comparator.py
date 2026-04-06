"""Comparison module: detected document data vs. user-declared form values.

Flags:
- income_mismatch   : >20% gap between declared and extracted income
- hidden_loans      : more active loans in credit report than declared
- high_dti          : extracted DTI exceeds RBI recommended 50% ceiling
"""
from __future__ import annotations

from typing import Any, Dict

# Thresholds
_INCOME_MISMATCH_THRESHOLD = 0.20   # 20% relative difference
_DTI_LIMIT = 0.50                   # RBI recommended ceiling


def compare_with_form(
    extracted: Dict[str, Any],
    form_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare extracted document data with the applicant's declared form values.

    Args:
        extracted:  Combined dict from all uploaded documents.
                    May contain: income, emi, dti, credit_score, active_loans.
        form_data:  Applicant declarations.
                    Expected keys: declared_income (float), declared_loans (int).

    Returns:
        {
            "mismatches":      dict of named mismatch entries,
            "has_mismatches":  bool,
            "mismatch_count":  int,
        }
    """
    mismatches: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Income mismatch
    # ------------------------------------------------------------------
    extracted_income = extracted.get("income")
    declared_income = form_data.get("declared_income")

    if extracted_income and declared_income:
        diff_ratio = abs(extracted_income - declared_income) / max(abs(declared_income), 1)
        if diff_ratio > _INCOME_MISMATCH_THRESHOLD:
            mismatches["income_mismatch"] = {
                "declared": declared_income,
                "extracted": extracted_income,
                "difference_pct": round(diff_ratio * 100, 2),
                "severity": "high" if diff_ratio > 0.40 else "medium",
            }

    # ------------------------------------------------------------------
    # 2. Hidden (undisclosed) loans
    # ------------------------------------------------------------------
    active_loans = extracted.get("active_loans")
    declared_loans = form_data.get("declared_loans")

    if active_loans is not None and declared_loans is not None:
        if active_loans > declared_loans:
            mismatches["hidden_loans"] = {
                "declared": declared_loans,
                "extracted": active_loans,
                "undisclosed_count": active_loans - declared_loans,
                "severity": "high",
            }

    # ------------------------------------------------------------------
    # 3. High DTI
    # ------------------------------------------------------------------
    extracted_dti = extracted.get("dti")
    if extracted_dti is not None and extracted_dti > _DTI_LIMIT:
        mismatches["high_dti"] = {
            "dti": extracted_dti,
            "rbi_limit": _DTI_LIMIT,
            "severity": "high" if extracted_dti > 0.65 else "medium",
        }

    # ------------------------------------------------------------------
    # 4. Low credit score (below common RBI / lender floor of 650)
    # ------------------------------------------------------------------
    credit_score = extracted.get("credit_score")
    if credit_score is not None and credit_score < 650:
        mismatches["low_credit_score"] = {
            "score": credit_score,
            "minimum_threshold": 650,
            "severity": "high" if credit_score < 550 else "medium",
        }

    return {
        "mismatches": mismatches,
        "has_mismatches": bool(mismatches),
        "mismatch_count": len(mismatches),
    }
