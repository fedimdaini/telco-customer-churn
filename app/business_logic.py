"""
Shared prediction and business-logic helpers used by both the Streamlit app
and the FastAPI service.  Keeps behaviour from silently diverging between
frontends.

All functions are stateless and accept their dependencies (preprocessor, clf,
explainer, metadata) as arguments so callers control caching / lifespan.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from app.churn_features import add_engineered_features

ARTIFACTS_DIR = Path("artifacts")

YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
]


# ----------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------
def load_artifacts():
    """Load preprocessor, model, metadata, and build SHAP explainer."""
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    clf = XGBClassifier()
    clf.load_model(ARTIFACTS_DIR / "xgb_model.json")
    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)
    explainer = shap.TreeExplainer(clf)
    return preprocessor, clf, explainer, metadata


# ----------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------
def encode_raw_inputs(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same Yes/No -> 1/0 and gender encoding used in training."""
    for col in YES_NO_COLS:
        if col in raw.columns and raw[col].dtype == object:
            raw[col] = raw[col].map({"Yes": 1, "No": 0})
    if "gender" in raw.columns and raw["gender"].dtype == object:
        raw["gender"] = raw["gender"].map({"Male": 1, "Female": 0})
    return raw


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def predict(raw_df: pd.DataFrame, preprocessor, clf):
    """raw_df: one row per customer, original (pre-engineering) columns."""
    features = add_engineered_features(raw_df)
    X_transformed = preprocessor.transform(features)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    proba = clf.predict_proba(X_transformed)[:, 1]
    return proba, X_transformed


# ----------------------------------------------------------------------
# SHAP
# ----------------------------------------------------------------------
def shap_explanation(X_transformed_row, explainer, preprocessor):
    feature_names = preprocessor.get_feature_names_out()
    shap_values = explainer.shap_values(X_transformed_row)
    assert shap_values.ndim == 2, f"Expected 2D shap_values, got {shap_values.ndim}D"
    sv = shap_values[0]
    contrib = pd.DataFrame({"feature": feature_names, "shap_value": sv})
    contrib = contrib.reindex(
        contrib.shap_value.abs().sort_values(ascending=False).index
    ).head(8)
    return contrib


# ----------------------------------------------------------------------
# Risk band
# ----------------------------------------------------------------------
def risk_band(proba: float, threshold: float):
    if proba >= threshold:
        return "High risk", "error"
    elif proba >= threshold * 0.6:
        return "Moderate risk", "warning"
    else:
        return "Low risk", "success"


# ----------------------------------------------------------------------
# Business-value framing helpers
# ----------------------------------------------------------------------
def _strip_prefix(feat: str) -> str:
    """Strip sklearn ColumnTransformer prefixes from feature names."""
    for prefix in ("onehot__", "scale__", "remainder__", "binary__",
                   "num__", "cat__", "passthrough__"):
        if feat.startswith(prefix):
            return feat[len(prefix):]
    return feat


def get_reasons_and_actions(contrib_df: pd.DataFrame, raw_row: dict = None, n_top: int = 3):
    """Map top SHAP features to human-readable risk factors and retention actions.

    Parameters
    ----------
    contrib_df : pd.DataFrame
        DataFrame with columns 'feature' and 'shap_value'.
    raw_row : dict, optional
        The original (already encoded) raw feature values for this customer.
        Needed for passthrough binary columns where the feature name does not
        encode the value (e.g. 'TechSupport' vs 'onehot__Contract_Month-to-month').
    n_top : int
        Number of top factors to explain.
    """

    ACTION_MAP = {
        "Contract_Month-to-month": (
            "Month-to-month contract (strongest churn predictor)",
            "Offer a one-year or two-year contract with a loyalty discount"
        ),
        "Contract_One year": (
            "One-year contract (lower risk than month-to-month)",
            "Maintain current contract terms; consider two-year upgrade at renewal"
        ),
        "Contract_Two year": (
            "Two-year contract (strong retention signal)",
            "Continue current contract; no immediate action needed"
        ),
        "charge_per_tenure_month": (
            "High charge-to-tenure ratio — customer may feel undervalued",
            "Offer a tenure-based loyalty discount or review plan fit"
        ),
        "InternetService_Fiber optic": (
            "Fiber optic service shows elevated churn risk",
            "Run a proactive service quality check or offer a bundled value-add"
        ),
        "InternetService_DSL": (
            "DSL service (moderate risk profile)",
            "Consider upgrade path to higher-tier plans with better value perception"
        ),
        "InternetService_No": (
            "No internet service (lower churn risk)",
            "Maintain current service package"
        ),
        "MonthlyCharges": (
            "High monthly bill increases churn pressure",
            "Review plan fit or offer a tailored downgrade path that preserves revenue"
        ),
        "PaymentMethod_Electronic check": (
            "Electronic check payment correlates with higher churn",
            "Suggest switching to automatic bank transfer or credit card with a small discount"
        ),
        "PaymentMethod_Mailed check": (
            "Mailed check payment (moderate risk)",
            "Encourage switch to automatic payment for convenience and retention"
        ),
        "PaymentMethod_Bank transfer (automatic)": (
            "Automatic bank transfer (positive retention signal)",
            "Maintain current payment setup"
        ),
        "PaymentMethod_Credit card (automatic)": (
            "Automatic credit card payment (positive retention signal)",
            "Maintain current payment setup"
        ),
        "tenure": (
            "Low tenure — relationship is not yet sticky",
            "Increase early-stage engagement with onboarding check-ins and welcome offers"
        ),
    }

    # For passthrough binary columns, the feature name does not encode the value.
    # We look up raw_row to know if the value is 0 or 1.
    BINARY_ACTIONS = {
        "TechSupport": {
            0: ("No tech support plan", "Offer tech support as a value-add or bundle inclusion"),
            1: ("Tech support plan active", "Maintain tech support inclusion"),
        },
        "OnlineSecurity": {
            0: ("No online security service", "Bundle online security at a promotional rate"),
            1: ("Online security active", "Maintain online security service"),
        },
        "OnlineBackup": {
            0: ("No online backup service", "Offer online backup as part of a retention bundle"),
            1: ("Online backup active", "Maintain online backup service"),
        },
        "DeviceProtection": {
            0: ("No device protection plan", "Suggest device protection to increase stickiness"),
            1: ("Device protection active", "Maintain device protection plan"),
        },
        "StreamingTV": {
            0: ("No streaming TV service", "Consider entertainment bundle to increase engagement"),
            1: ("Streaming TV active", "Maintain streaming TV service"),
        },
        "StreamingMovies": {
            0: ("No streaming movies service", "Consider entertainment bundle to increase engagement"),
            1: ("Streaming movies active", "Maintain streaming movies service"),
        },
        "PaperlessBilling": {
            0: ("Paperless billing not enabled", "Offer paperless billing with a small incentive"),
            1: ("Paperless billing enabled", "Maintain paperless billing setup"),
        },
        "Partner": {
            0: ("No partner on account", "Offer family plan or referral incentives"),
            1: ("Partner on account", "Maintain family-friendly plan options"),
        },
        "Dependents": {
            0: ("No dependents on account", "Consider individual-focused retention offers"),
            1: ("Dependents on account", "Offer family plan upgrades or child-friendly bundles"),
        },
        "PhoneService": {
            0: ("No phone service", "Consider voice bundle to increase stickiness"),
            1: ("Phone service active", "Maintain phone service plan"),
        },
        "MultipleLines": {
            0: ("Single line only", "Offer multiple lines with a bundle discount"),
            1: ("Multiple lines active", "Maintain multi-line discount structure"),
        },
        "SeniorCitizen": {
            0: ("Non-senior profile", "Review plan fit for current life stage"),
            1: ("Senior citizen profile", "Ensure plan meets accessibility and value needs"),
        },
    }

    reasons = []
    actions = []

    for _, row in contrib_df.head(n_top).iterrows():
        feat = row["feature"]
        val = row["shap_value"]
        base_feat = _strip_prefix(feat)

        if base_feat in ACTION_MAP:
            reason, action = ACTION_MAP[base_feat]
        elif base_feat in BINARY_ACTIONS:
            raw_val = raw_row.get(base_feat, 0) if raw_row is not None else 0
            reason, action = BINARY_ACTIONS[base_feat].get(
                raw_val,
                (f"{base_feat} status unknown", "Review customer profile")
            )
        elif "gender" in base_feat:
            raw_val = raw_row.get("gender", 0) if raw_row is not None else 0
            if raw_val == 1:
                reason = "Male customer profile"
                action = "Review plan fit for current profile"
            else:
                reason = "Female customer profile"
                action = "Review plan fit for current profile"
        else:
            reason = f"{base_feat} contributes to this risk profile"
            action = "Review customer profile for tailored retention approach"

        direction = "increases" if val > 0 else "decreases"
        reasons.append(f"• **{reason}** — {direction} churn risk (SHAP: {val:+.3f})")
        actions.append(f"• **{action}**")

    return reasons, actions