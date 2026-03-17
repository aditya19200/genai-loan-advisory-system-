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
        df["saving_accounts"] = df["saving_accounts"].fillna("unknown")
        df["checking_account"] = df["checking_account"].fillna("unknown")
        df["sex"] = df["sex"].str.lower()
        df["housing"] = df["housing"].str.lower()
        df["purpose"] = df["purpose"].str.lower()
        return self.engineer_features(df)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["monthly_burden"] = df["credit_amount"] / df["duration"].clip(lower=1)
        df["credit_to_age"] = df["credit_amount"] / df["age"].clip(lower=18)
        df["age_group"] = np.where(df["age"] >= 30, "age_30_plus", "under_30")
        df["has_savings_buffer"] = df["saving_accounts"].isin(["moderate", "rich", "quite rich"]).astype(int)
        df["has_checking_buffer"] = df["checking_account"].isin(["moderate", "rich"]).astype(int)
        df["long_duration"] = (df["duration"] >= 24).astype(int)
        df[TARGET_COLUMN] = self._derive_target(df)
        return df

    def _derive_target(self, df: pd.DataFrame) -> pd.Series:
        burden_score = (df["monthly_burden"] > df["monthly_burden"].median()).astype(int)
        amount_score = (df["credit_amount"] > df["credit_amount"].median()).astype(int)
        duration_score = (df["duration"] > df["duration"].median()).astype(int)
        account_risk = (
            (df["saving_accounts"].isin(["little", "unknown"])).astype(int)
            + (df["checking_account"].isin(["little", "unknown"])).astype(int)
        )
        housing_risk = (df["housing"] == "rent").astype(int)
        purpose_risk = df["purpose"].isin(["education", "business", "vacation/others"]).astype(int)
        score = burden_score + amount_score + duration_score + account_risk + housing_risk + purpose_risk
        return (score >= 4).astype(int)

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
        X = df.drop(columns=[TARGET_COLUMN])
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
            feature_columns=X_train.columns.tolist(),
        )

    @staticmethod
    def api_payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([payload])
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
        df["saving_accounts"] = df["saving_accounts"].fillna("unknown")
        df["checking_account"] = df["checking_account"].fillna("unknown")
        df["sex"] = df["sex"].str.lower()
        df["housing"] = df["housing"].str.lower()
        df["purpose"] = df["purpose"].str.lower()
        df["monthly_burden"] = df["credit_amount"] / df["duration"].clip(lower=1)
        df["credit_to_age"] = df["credit_amount"] / df["age"].clip(lower=18)
        df["age_group"] = np.where(df["age"] >= 30, "age_30_plus", "under_30")
        df["has_savings_buffer"] = df["saving_accounts"].isin(["moderate", "rich", "quite rich"]).astype(int)
        df["has_checking_buffer"] = df["checking_account"].isin(["moderate", "rich"]).astype(int)
        df["long_duration"] = (df["duration"] >= 24).astype(int)
        return df
