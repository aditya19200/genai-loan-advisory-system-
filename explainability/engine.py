from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from app.config import BACKGROUND_SAMPLE_SIZE, TOP_K_EXPLANATION_FEATURES
from explainability.rules import RuleBasedExplainer


class ExplainabilityEngine:
    def __init__(self, bundle: dict[str, Any]) -> None:
        self.pipeline = bundle["model"]
        self.train_frame: pd.DataFrame = bundle["train_frame"]
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.transformed_train = self.preprocessor.transform(self.train_frame)
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        background_rows = min(BACKGROUND_SAMPLE_SIZE, len(self.train_frame))
        self.background = shap.sample(self._dense(self.transformed_train), background_rows, random_state=42)
        self.dense_train = self._dense(self.transformed_train)
        self.rule_explainer = RuleBasedExplainer(bundle["train_frame"], bundle["train_target"])
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.dense_train,
            feature_names=self.feature_names,
            class_names=["low_risk", "high_risk"],
            mode="classification",
            discretize_continuous=True,
        )

    def _shap_explainer(self):
        classifier_name = self.classifier.__class__.__name__.lower()
        classifier_module = self.classifier.__class__.__module__.lower()
        if hasattr(self.classifier, "estimators_") or "xgboost" in classifier_module or classifier_name.startswith("xgb"):
            return shap.TreeExplainer(self.classifier)
        if classifier_name.startswith("logistic"):
            return shap.LinearExplainer(self.classifier, self.background)
        return shap.Explainer(self.classifier.predict_proba, self.background)

    def explain(self, input_frame: pd.DataFrame) -> dict[str, Any]:
        transformed_input = self.preprocessor.transform(input_frame)
        dense_input = self._dense(transformed_input)
        shap_explainer = self._shap_explainer()
        shap_values = shap_explainer(dense_input)
        positive_index = 1 if len(getattr(shap_values, "values", np.array([])).shape) == 3 else None
        shap_vector = shap_values.values[0, :, positive_index] if positive_index is not None else shap_values.values[0]
        local_scores = self._top_feature_pairs(self.feature_names, shap_vector, absolute=True)
        lime_exp = self.lime_explainer.explain_instance(
            dense_input[0],
            self.classifier.predict_proba,
            num_features=TOP_K_EXPLANATION_FEATURES,
        )
        lime_pairs = [{"feature": name, "importance": float(score)} for name, score in lime_exp.as_list()]
        eli5_text = self._readable_summary(local_scores, lime_pairs)
        model_prediction = int(self.pipeline.predict(input_frame)[0])
        return {
            "shap_global": self._global_importance(),
            "shap_local": local_scores,
            "lime_local": lime_pairs,
            "rule_based_explanation": self.rule_explainer.explain_case(input_frame, model_prediction),
            "eli5_summary": eli5_text,
        }

    def _global_importance(self) -> list[dict[str, Any]]:
        sample = self._dense(self.transformed_train)[:BACKGROUND_SAMPLE_SIZE]
        shap_explainer = self._shap_explainer()
        shap_values = shap_explainer(sample)
        values = shap_values.values
        if values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        return self._top_feature_pairs(self.feature_names, mean_abs, absolute=False)

    @staticmethod
    def _top_feature_pairs(feature_names: list[str], values: np.ndarray, absolute: bool) -> list[dict[str, Any]]:
        scores = np.abs(values) if absolute else values
        top_idx = np.argsort(scores)[::-1][:TOP_K_EXPLANATION_FEATURES]
        return [{"feature": feature_names[idx], "importance": float(values[idx])} for idx in top_idx]

    @staticmethod
    def stability_score(shap_local: list[dict[str, Any]], lime_local: list[dict[str, Any]]) -> float:
        shap_set = {ExplainabilityEngine._normalize_feature_name(item["feature"]) for item in shap_local}
        lime_set = {ExplainabilityEngine._normalize_feature_name(item["feature"]) for item in lime_local}
        union = shap_set | lime_set
        if not union:
            return 1.0
        return round(len(shap_set & lime_set) / len(union), 4)

    @staticmethod
    def _dense(matrix) -> np.ndarray:
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    @staticmethod
    def _normalize_feature_name(feature: str) -> str:
        value = feature.replace("num__", "").replace("cat__", "")
        value = re.sub(r"\b(little|moderate|rich|unknown|male|female|own|rent|free)\b", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<=|>=|=|<|>", " ", value)
        value = re.sub(r"-?\d+(\.\d+)?", " ", value)
        value = value.replace("_", " ")
        value = re.sub(r"\s+", " ", value).strip().lower()
        return value

    @staticmethod
    def _humanize_feature_name(feature: str) -> str:
        value = feature.replace("num__", "").replace("cat__", "")
        value = re.sub(r"\s+", " ", value.replace("_", " ")).strip()
        if any(op in value for op in ["<=", ">=", "<", ">"]):
            return value.title()
        if " " in value:
            head, tail = value.rsplit(" ", 1)
            categories = {"male", "female", "own", "rent", "free", "little", "moderate", "rich", "unknown"}
            if tail.lower() in categories:
                return f"{head.title()}: {tail.title()}"
        return value.title()

    def _readable_summary(self, shap_local: list[dict[str, Any]], lime_local: list[dict[str, Any]]) -> str:
        top_shap = [
            f"{self._humanize_feature_name(item['feature'])} ({'increased' if item['importance'] > 0 else 'reduced'} risk)"
            for item in shap_local[:3]
        ]
        top_lime = [self._humanize_feature_name(item["feature"]) for item in lime_local[:3]]
        summary_lines = [
            "This explanation highlights the main factors that pushed the model toward its decision.",
            "",
            "Top SHAP drivers:",
            *[f"- {item}" for item in top_shap],
            "",
            "Top LIME factors:",
            *[f"- {item}" for item in top_lime],
            "",
            "Use SHAP to understand contribution strength and LIME to understand the local rule-like explanation around this case.",
        ]
        return "\n".join(summary_lines)
