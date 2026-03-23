"""Data preparation for the German credit dataset.

The raw dataset does not contain the exact fields used by the API contract
(`income`, `credit_score`, `dti`, `employment_length`, `existing_loans`), so
this module derives a stable training set from the available columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.processor import FEATURE_ORDER

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "Data" / "german_credit_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "Data" / "processed_loan_data.csv"

SAVINGS_FACTOR = {
    "none": 0.0,
    "little": 0.4,
    "moderate": 0.8,
    "quite rich": 1.2,
    "rich": 1.6,
}

CHECKING_FACTOR = {
    "none": 0.0,
    "little": 0.6,
    "moderate": 1.0,
    "rich": 1.4,
}

HOUSING_FACTOR = {
    "free": -20.0,
    "rent": 0.0,
    "own": 20.0,
}


def _clean_account_value(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"nan", "na", "none", ""}:
        return "none"
    return text


def process_german_data(
    raw_path: str | Path = RAW_DATA_PATH,
    processed_path: str | Path = PROCESSED_DATA_PATH,
) -> Path:
    """Create a model-ready dataset aligned with the serving feature schema."""
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}")

    df = pd.read_csv(raw_path, index_col=0)

    df["Saving accounts"] = df["Saving accounts"].map(_clean_account_value)
    df["Checking account"] = df["Checking account"].map(_clean_account_value)
    df["Housing"] = df["Housing"].astype(str).str.strip().str.lower()

    savings_score = df["Saving accounts"].map(SAVINGS_FACTOR).fillna(0.0)
    checking_score = df["Checking account"].map(CHECKING_FACTOR).fillna(0.0)
    housing_score = df["Housing"].map(HOUSING_FACTOR).fillna(0.0)

    income = (
        df["Credit amount"] * (2.2 + df["Job"] * 0.45)
        + df["Age"] * 35
        + savings_score * 400
    )
    income = income.clip(lower=1200).round(2)

    credit_score = (
        550
        + df["Age"] * 1.2
        + df["Job"] * 25
        + savings_score * 60
        + checking_score * 40
        + housing_score
        - df["Duration"] * 1.5
        - df["Credit amount"] / 120
    )
    credit_score = credit_score.clip(lower=300, upper=850).round(0)

    dti = (df["Credit amount"] / income).clip(lower=0.01, upper=1.5).round(4)
    employment_length = (df["Age"] - 18).clip(lower=0, upper=45).round(1)
    existing_loans = (
        (df["Duration"] / 24).round()
        + (df["Credit amount"] >= 5000).astype(int)
        + df["Checking account"].isin({"none", "little"}).astype(int)
    ).astype(float)

    risk_score = (
        dti * 4.0
        + (df["Duration"] / 12) * 0.7
        + (df["Credit amount"] / 1000) * 0.25
        + (700 - credit_score) / 75
        + df["Checking account"].eq("little").astype(int) * 0.7
        + df["Saving accounts"].eq("little").astype(int) * 0.5
    )
    threshold = np.percentile(risk_score, 70)
    risk = (risk_score >= threshold).astype(int)

    processed = pd.DataFrame(
        {
            "income": income,
            "credit_score": credit_score,
            "dti": dti,
            "employment_length": employment_length,
            "existing_loans": existing_loans,
            "Risk": risk,
        }
    )

    processed = processed[FEATURE_ORDER + ["Risk"]]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    return processed_path


if __name__ == "__main__":
    output_path = process_german_data()
    print(f"Processed dataset written to {output_path}")
