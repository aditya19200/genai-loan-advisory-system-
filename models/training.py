from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from app.config import METRICS_DIR, MODELS_DIR, TOP_MODEL_COUNT
from data.processing import DataProcessor
from models.evaluation import evaluate_model
from models.registry import ModelRegistry

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


@dataclass
class TrainedModelBundle:
    name: str
    version: str
    artifact_path: str
    metrics: dict[str, Any]
    estimator: Any


class ModelTrainer:
    def __init__(self) -> None:
        self.processor = DataProcessor()
        self.registry = ModelRegistry()

    def _candidate_configs(self) -> list[tuple[str, Any, dict[str, list[Any]]]]:
        configs: list[tuple[str, Any, dict[str, list[Any]]]] = [
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, solver="liblinear"),
                {
                    "classifier__C": [0.1, 1.0, 5.0],
                    "classifier__class_weight": [None, "balanced"],
                },
            ),
            (
                "random_forest",
                RandomForestClassifier(random_state=42),
                {
                    "classifier__n_estimators": [150, 250],
                    "classifier__max_depth": [4, 8, None],
                    "classifier__min_samples_split": [2, 5],
                },
            ),
        ]
        if XGBClassifier is not None:
            configs.append(
                (
                    "xgboost",
                    XGBClassifier(
                        eval_metric="logloss",
                        random_state=42,
                        n_estimators=150,
                        learning_rate=0.08,
                        max_depth=4,
                    ),
                    {
                        "classifier__n_estimators": [100, 150],
                        "classifier__max_depth": [3, 4, 5],
                        "classifier__learning_rate": [0.05, 0.08],
                    },
                )
            )
        return configs

    def train_all(self) -> list[TrainedModelBundle]:
        data = self.processor.prepare_data()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        bundles: list[TrainedModelBundle] = []
        self.registry.save({"models": []})

        for model_name, estimator, param_grid in self._candidate_configs():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", data.preprocessor),
                    ("classifier", estimator),
                ]
            )
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
            search.fit(data.X_train, data.y_train)
            best_model = search.best_estimator_
            metrics = evaluate_model(best_model, data.X_test, data.y_test, data.sensitive_test)
            version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            artifact_name = f"{model_name}_{version}.joblib"
            artifact_path = MODELS_DIR / artifact_name
            metadata_path = METRICS_DIR / f"{model_name}_{version}_metrics.json"
            joblib.dump(
                {
                    "model": best_model,
                    "feature_columns": data.feature_columns,
                    "train_frame": data.X_train,
                    "train_target": data.y_train,
                },
                artifact_path,
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "version": version,
                        "best_params": search.best_params_,
                        "cv_best_score": float(search.best_score_),
                        "metrics": metrics,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            bundles.append(
                TrainedModelBundle(
                    name=model_name,
                    version=version,
                    artifact_path=str(artifact_path),
                    metrics=metrics,
                    estimator=best_model,
                )
            )

        bundles.sort(key=lambda bundle: bundle.metrics["roc_auc"], reverse=True)
        for rank, bundle in enumerate(bundles[:TOP_MODEL_COUNT], start=1):
            self.registry.register(
                {
                    "name": bundle.name,
                    "version": bundle.version,
                    "artifact_path": bundle.artifact_path,
                    "metrics": bundle.metrics,
                    "rank": rank,
                }
            )
        return bundles

    def ensure_trained(self) -> list[dict[str, Any]]:
        top_models = self.registry.top_models(limit=TOP_MODEL_COUNT)
        if top_models:
            return top_models
        self.train_all()
        return self.registry.top_models(limit=TOP_MODEL_COUNT)


def load_registered_model(record: dict[str, Any]) -> dict[str, Any]:
    return joblib.load(Path(record["artifact_path"]))
