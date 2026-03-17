from __future__ import annotations

from typing import Any

import joblib
import numpy as np

try:
    from alibi.explainers import Counterfactual
except ImportError:  # pragma: no cover
    Counterfactual = None

from data.processing import DataProcessor


class AlibiCounterfactualService:
    def generate(self, input_payload: dict[str, Any], model_artifact_path: str) -> dict[str, Any]:
        if Counterfactual is None:
            return {
                "status": "unavailable",
                "message": "Alibi is not installed in this environment.",
                "suggested_changes": [],
            }
        bundle = joblib.load(model_artifact_path)
        pipeline = bundle["model"]
        train_frame = bundle["train_frame"]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        input_frame = DataProcessor.api_payload_to_frame(input_payload)
        dense_train = self._dense(preprocessor.transform(train_frame))
        dense_input = self._dense(preprocessor.transform(input_frame))
        feature_names = list(preprocessor.get_feature_names_out())
        feature_min = dense_train.min(axis=0)
        feature_max = dense_train.max(axis=0)
        try:
            explainer = Counterfactual(
                classifier.predict_proba,
                shape=dense_input.shape,
                target_proba=0.6,
                target_class="other",
                max_iter=500,
                lam_init=0.1,
                max_lam_steps=10,
                learning_rate_init=0.05,
                feature_range=(feature_min, feature_max),
            )
            explanation = explainer.explain(dense_input)
            cf_data = explanation.data.get("cf") if hasattr(explanation, "data") else None
            if cf_data is None or cf_data.get("X") is None:
                return {
                    "status": "not_found",
                    "message": "No counterfactual was found for this case.",
                    "suggested_changes": [],
                }
            counterfactual_point = cf_data["X"][0]
            delta = counterfactual_point - dense_input[0]
            suggestions = self._top_changes(feature_names, delta)
            return {
                "status": "ready",
                "message": "Counterfactual explanation generated successfully.",
                "suggested_changes": suggestions,
            }
        except Exception as exc:
            return {
                "status": "error",
                "message": f"Counterfactual generation failed: {exc}",
                "suggested_changes": [],
            }

    @staticmethod
    def _dense(matrix) -> np.ndarray:
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

    @staticmethod
    def _top_changes(feature_names: list[str], deltas: np.ndarray) -> list[dict[str, Any]]:
        ranked = np.argsort(np.abs(deltas))[::-1][:5]
        results = []
        for idx in ranked:
            direction = "increase" if deltas[idx] > 0 else "decrease"
            results.append(
                {
                    "feature": feature_names[idx],
                    "direction": direction,
                    "change_magnitude": float(deltas[idx]),
                }
            )
        return results
