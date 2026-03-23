"""ML training and inference services.

Responsibilities:
- train_xgboost: train and persist model + preprocessing pipeline
- load_model / load_preprocessor
- predict_single: run inference and return decision + risk score
- explain_shap: compute SHAP explanation for a single sample
"""
from __future__ import annotations

import os
import joblib
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Default paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_model.pkl")
DEFAULT_PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

def train_xgboost(
    data_csv: str,
    feature_cols: list[str],
    target_col: str = "Risk",
    model_out_path: str | None = None,
    preprocessor_out_path: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train an XGBoost classifier and save model + preprocessor pipeline."""
    
    model_out_path = model_out_path or DEFAULT_MODEL_PATH
    preprocessor_out_path = preprocessor_out_path or DEFAULT_PREPROCESSOR_PATH

    # 1. Load Data
    df = pd.read_csv(data_csv)
    X = df[feature_cols]
    y = df[target_col]

    # 2. Define Preprocessing Pipeline
    # We use a simple pipeline: Impute missing values then Scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, feature_cols),
    ])

    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Transform Data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    # 5. Train the decision model.
    if XGBClassifier is not None:
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
        )
    model.fit(X_train_proc, y_train)

    # 6. Evaluation
    from sklearn.metrics import roc_auc_score, accuracy_score
    val_proba = model.predict_proba(X_val_proc)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    metrics = {
        "val_auc": float(roc_auc_score(y_val, val_proba)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
    }

    # 7. Persist Artifacts
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(model, model_out_path)
    joblib.dump(preprocessor, preprocessor_out_path)

    return {
        "model_path": model_out_path,
        "preprocessor_path": preprocessor_out_path,
        "metrics": metrics,
        "model_type": model.__class__.__name__,
    }

def load_model(model_path: str | None = None):
    return joblib.load(model_path or DEFAULT_MODEL_PATH)

def load_preprocessor(preprocessor_path: str | None = None):
    return joblib.load(preprocessor_path or DEFAULT_PREPROCESSOR_PATH)

def predict_single(model, preprocessor, sample: Dict[str, Any], feature_order: list[str]) -> Tuple[str, float]:
    """Run inference for a single user sample."""
    # Convert dict to DataFrame with correct column order
    df = pd.DataFrame([sample])[feature_order]
    
    # Apply the same scaling/imputation used in training
    X_proc = preprocessor.transform(df)
    
    # Get probability of class 1 (Rejected)
    proba = float(model.predict_proba(X_proc)[0, 1])

    # Decision logic: 1 = Rejected, 0 = Approved
    decision = "Rejected" if proba >= 0.5 else "Approved"
    
    return decision, proba

def explain_shap(
    model,
    preprocessor,
    sample: Dict[str, Any],
    feature_order: list[str],
    background_df: pd.DataFrame | None = None,
) -> Dict[str, float]:
    """Compute SHAP values for a single sample.
    
    Positive SHAP values = Increase Risk (Push towards Rejection)
    Negative SHAP values = Decrease Risk (Push towards Approval)
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install the explainability dependencies first.") from exc

    # 1. Prepare the sample
    df_sample = pd.DataFrame([sample])[feature_order]
    X_sample_proc = preprocessor.transform(df_sample)
    if background_df is None:
        background_df = pd.DataFrame([{feature: 0.0 for feature in feature_order}])
    X_background_proc = preprocessor.transform(background_df[feature_order])

    # 2. Initialize Explainer
    if XGBClassifier is not None and isinstance(model, XGBClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_proc)
    else:
        explainer = shap.Explainer(model.predict_proba, X_background_proc)
        shap_values = explainer(X_sample_proc).values

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    return dict(zip(feature_order, [float(x) for x in sv]))
