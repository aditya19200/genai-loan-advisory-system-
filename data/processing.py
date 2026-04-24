from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import DATASET_PATH, TRAIN_TEST_RANDOM_STATE


TARGET_COLUMN = "credit_risk"
MODEL_FEATURES = ["age", "income", "credit_score", "dti", "employment_length", "existing_loans"]
ADVISORY_FEATURES = ["loan_amount", "tenure_months"]
SENSITIVE_ATTRIBUTES = ["Sex", "age_group"]


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sensitive_train: pd.DataFrame
    sensitive_test: pd.DataFrame
    preprocessor: ColumnTransformer
    feature_columns: list[str]


class DataProcessor:
    def __init__(self, csv_path: str | None = None) -> None:
        self.csv_path = csv_path or str(DATASET_PATH)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        return self.clean_data(df)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        unnamed = [col for col in df.columns if col.startswith("Unnamed") or col == ""]
        if unnamed:
            df = df.drop(columns=unnamed)
        df = df.replace({"NA": np.nan})
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
        return self.engineer_features(df)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["saving_accounts"] = df["saving_accounts"].fillna("unknown").str.lower()
        df["checking_account"] = df["checking_account"].fillna("unknown").str.lower()
        df["sex"] = df["sex"].str.lower()
        df["housing"] = df["housing"].str.lower()
        df["purpose"] = df["purpose"].str.lower()
        df["age_group"] = np.where(df["age"] >= 30, "age_30_plus", "under_30")
        savings_score = df["saving_accounts"].map(
            {"unknown": 0.2, "little": 0.4, "moderate": 0.8, "rich": 1.4, "quite rich": 1.8}
        ).fillna(0.2)
        checking_score = df["checking_account"].map(
            {"unknown": 0.2, "little": 0.6, "moderate": 1.0, "rich": 1.4}
        ).fillna(0.2)
        housing_score = df["housing"].map({"free": -15.0, "rent": 0.0, "own": 20.0}).fillna(0.0)

        income = (
            df["credit_amount"] * (2.2 + df["job"] * 0.45)
            + df["age"] * 35
            + savings_score * 400
        ).clip(lower=1200)
        credit_score = (
            550
            + df["age"] * 1.2
            + df["job"] * 25
            + savings_score * 60
            + checking_score * 40
            + housing_score
            - df["duration"] * 1.5
            - df["credit_amount"] / 120
        ).clip(lower=300, upper=850)
        dti = (df["credit_amount"] / income).clip(lower=0.01, upper=1.5)
        employment_length = (df["age"] - 18).clip(lower=0, upper=45)
        existing_loans = (
            (df["duration"] / 24).round()
            + (df["credit_amount"] >= 5000).astype(int)
            + df["checking_account"].isin({"unknown", "little"}).astype(int)
        ).astype(float)

        df["income"] = income.round(2)
        df["credit_score"] = credit_score.round(0)
        df["dti"] = dti.round(4)
        df["employment_length"] = employment_length.round(1)
        df["existing_loans"] = existing_loans
        df["loan_amount"] = df["credit_amount"].astype(float)
        df["tenure_months"] = df["duration"].astype(int)
        df[TARGET_COLUMN] = self._derive_target(df)
        return df

    def _derive_target(self, df: pd.DataFrame) -> pd.Series:
        risk_score = (
            df["dti"] * 4.0
            + (df["tenure_months"] / 12) * 0.7
            + (df["loan_amount"] / 1000) * 0.25
            + (700 - df["credit_score"]) / 75
            + df["checking_account"].eq("little").astype(int) * 0.7
            + df["saving_accounts"].eq("little").astype(int) * 0.5
        )
        threshold = np.percentile(risk_score, 70)
        return (risk_score >= threshold).astype(int)

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_features = [col for col in X.columns if col not in categorical_features]
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ]
        )

    def prepare_data(self, test_size: float = 0.2) -> PreparedData:
        df = self.load_data()
        X = df[MODEL_FEATURES]
        y = df[TARGET_COLUMN]
        sensitive = df[["sex", "age_group"]].rename(columns={"sex": "Sex"})
        X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
            X,
            y,
            sensitive,
            test_size=test_size,
            stratify=y,
            random_state=TRAIN_TEST_RANDOM_STATE,
        )
        preprocessor = self.build_preprocessor(X_train)
        return PreparedData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            sensitive_train=sensitive_train,
            sensitive_test=sensitive_test,
            preprocessor=preprocessor,
            feature_columns=MODEL_FEATURES.copy(),
        )

    @staticmethod
    def api_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
        normalized = {
            "age": int(payload["age"]),
            "income": float(payload["income"]),
            "credit_score": float(payload["credit_score"]),
            "dti": float(payload["dti"]),
            "employment_length": float(payload["employment_length"]),
            "existing_loans": float(payload["existing_loans"]),
        }
        return pd.DataFrame([normalized], columns=MODEL_FEATURES)
