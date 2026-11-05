"""
Telco Customer Churn Predictor -- Streamlit app.

Two modes:
  1. Single customer lookup -- enter one customer's details, get a churn
     probability + a per-customer SHAP explanation of why.
  2. Batch ranking -- upload a CSV of customers, get them back ranked by
     churn probability. This is the mode that matches the actual business
     recommendation from the project (rank, don't rely on one fixed
     threshold) rather than just a single-prediction demo.

Run with: streamlit run app/app.py
(run from the project root so the relative artifact paths resolve)
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from xgboost import XGBClassifier

from app.churn_features import add_engineered_features
from app.business_logic import (
    encode_raw_inputs,
    get_reasons_and_actions,
    load_artifacts,
    predict,
    risk_band,
    shap_explanation,
)

st.set_page_config(page_title="Telco Churn Predictor", page_icon="\U0001F4C9", layout="wide")

ARTIFACTS_DIR = "artifacts"


# ----------------------------------------------------------------------
# Cached loaders -- run once, reused across every rerun of the script
# ----------------------------------------------------------------------
@st.cache_resource
def _load_all():
    return load_artifacts()


preprocessor, clf, explainer, metadata = _load_all()

FINAL_THRESHOLD = metadata["final_threshold"]
NUMERIC_COLS = metadata["numeric_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
BINARY_COLS = metadata["binary_cols"]

# charge_per_tenure_month is computed, not entered by the user --
# exclude it from the raw numeric input fields.
RAW_NUMERIC_COLS = [c for c in NUMERIC_COLS if c != "charge_per_tenure_month"]

# Options for each categorical/binary field. Hardcoded here since these
# are fixed by the dataset's schema, not derived from metadata.json --
# metadata.json tells us WHICH columns exist, not what values they take.
CATEGORY_OPTIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"],
    "InternetService": ["DSL", "Fiber optic", "No"],
}
YES_NO_COLS = [c for c in BINARY_COLS if c not in ("gender", "SeniorCitizen")]


# ----------------------------------------------------------------------
# Shared prediction logic -- used by both modes, so behavior can't
# silently diverge between the single-customer and batch paths.
# ----------------------------------------------------------------------
def _predict(raw_df: pd.DataFrame):
    """Wrapper that binds the loaded artifacts to the shared predict()."""
    return predict(raw_df, preprocessor, clf)


def _shap_explanation(X_transformed_row):
    """Wrapper that binds loaded artifacts to the shared shap_explanation()."""
    return shap_explanation(X_transformed_row, explainer, preprocessor)


# ----------------------------------------------------------------------
# Page layout
# ----------------------------------------------------------------------
st.title("\U0001F4C9 Telco Customer Churn Predictor")
st.caption(
    "Predicts churn probability and explains why, so a retention team can "
    "prioritize outreach before customers leave."
)

mode = st.sidebar.radio("Mode", ["Single customer", "Batch ranking"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model:** XGBoost  \n"
    f"**Test ROC-AUC:** {metadata['test_roc_auc']:.3f}  \n"
    f"**Fallback threshold:** {FINAL_THRESHOLD:.2f}  \n\n"
    "Recommended use: rank customers by probability and work down the list "
    "to match retention team capacity, rather than relying on one fixed "
    "cutoff (see project README for why)."
)

# ========================================================================
# MODE 1: Single customer
# ========================================================================
if mode == "Single customer":
    st.markdown("### Enter customer details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", CATEGORY_OPTIONS["Contract"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", CATEGORY_OPTIONS["PaymentMethod"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            value=float(monthly_charges * max(tenure, 1)),
        )

    with col2:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

    with col3:
        st.markdown("**Internet & add-ons**")
        internet = st.selectbox("Internet Service", CATEGORY_OPTIONS["InternetService"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

    if st.button("Predict churn risk", type="primary"):
        raw = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }])

        raw = encode_raw_inputs(raw)

        proba, X_transformed = _predict(raw)
        proba = proba[0]

        # Store baseline for what-if simulator
        st.session_state["baseline_proba"] = float(proba)
        st.session_state["baseline_contrib"] = _shap_explanation(X_transformed)
        st.session_state["baseline_raw"] = raw.copy()

        st.markdown("---")
        r1, r2 = st.columns([1, 2])

        with r1:
            st.metric("Predicted churn probability", f"{proba:.0%}")
            label, kind = risk_band(proba, FINAL_THRESHOLD)
            getattr(st, kind)(f"{label} (fallback threshold: {FINAL_THRESHOLD:.0%})")

            # Business-value framing: top reasons and recommended actions
            baseline_raw_row = st.session_state["baseline_raw"].iloc[0].to_dict()
            reasons, actions = get_reasons_and_actions(
                st.session_state["baseline_contrib"], raw_row=baseline_raw_row, n_top=3
            )

            st.markdown("**Top risk factors**")
            for r in reasons:
                st.markdown(r)

            st.markdown("**Recommended retention actions**")
            for a in actions:
                st.markdown(a)

        with r2:
            st.markdown("**Why this prediction? (SHAP explanation)**")
            contrib = st.session_state["baseline_contrib"]

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#c0392b" if v > 0 else "#2980b9" for v in contrib["shap_value"]]
            ax.barh(contrib["feature"], contrib["shap_value"], color=colors)
            ax.set_xlabel("SHAP value (impact on churn probability)")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close(fig)

    # ----------------------------------------------------------------------
    # What-If Simulator
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔮 What-If Simulator")

    if "baseline_proba" not in st.session_state:
        st.info(
            "Run a prediction above to establish a baseline, then use this simulator "
            "to explore how changing key attributes affects churn risk."
        )
    else:
        st.markdown(
            "Adjust the levers below to simulate a different customer profile. "
            "The simulator reuses the same prediction pipeline as the main model."
        )

        base_raw = st.session_state["baseline_raw"]

        def _yn_idx(col_name):
            """Return selectbox index for a binary column stored as 0/1."""
            val = base_raw[col_name].iloc[0]
            return 0 if val == 1 else 1  # Yes=0, No=1 in ["Yes", "No"]

        wi_col1, wi_col2, wi_col3 = st.columns(3)

        with wi_col1:
            wi_contract = st.selectbox(
                "Contract",
                CATEGORY_OPTIONS["Contract"],
                index=CATEGORY_OPTIONS["Contract"].index(base_raw["Contract"].iloc[0]),
                key="wi_contract",
            )
            wi_payment = st.selectbox(
                "Payment Method",
                CATEGORY_OPTIONS["PaymentMethod"],
                index=CATEGORY_OPTIONS["PaymentMethod"].index(base_raw["PaymentMethod"].iloc[0]),
                key="wi_payment",
            )
            wi_internet = st.selectbox(
                "Internet Service",
                CATEGORY_OPTIONS["InternetService"],
                index=CATEGORY_OPTIONS["InternetService"].index(base_raw["InternetService"].iloc[0]),
                key="wi_internet",
            )

        with wi_col2:
            wi_monthly = st.slider(
                "Monthly Charges ($)", 18.0, 120.0,
                float(base_raw["MonthlyCharges"].iloc[0]),
                key="wi_monthly",
            )
            wi_tenure = st.slider(
                "Tenure (months)", 0, 72,
                int(base_raw["tenure"].iloc[0]),
                key="wi_tenure",
            )
            wi_total = st.number_input(
                "Total Charges ($)", min_value=0.0,
                value=float(wi_monthly * max(wi_tenure, 1)),
                key="wi_total",
            )

        with wi_col3:
            wi_security = st.selectbox(
                "Online Security", ["Yes", "No"],
                index=_yn_idx("OnlineSecurity"),
                key="wi_security",
            )
            wi_tech = st.selectbox(
                "Tech Support", ["Yes", "No"],
                index=_yn_idx("TechSupport"),
                key="wi_tech",
            )
            wi_backup = st.selectbox(
                "Online Backup", ["Yes", "No"],
                index=_yn_idx("OnlineBackup"),
                key="wi_backup",
            )

        if st.button("Run simulation", key="wi_run"):
            # Build what-if dataframe, preserving unchanged fields from baseline
            wi_raw = base_raw.copy()
            wi_raw["Contract"] = wi_contract
            wi_raw["PaymentMethod"] = wi_payment
            wi_raw["InternetService"] = wi_internet
            wi_raw["MonthlyCharges"] = wi_monthly
            wi_raw["tenure"] = wi_tenure
            wi_raw["TotalCharges"] = wi_total
            wi_raw["OnlineSecurity"] = 1 if wi_security == "Yes" else 0
            wi_raw["OnlineBackup"] = 1 if wi_backup == "Yes" else 0
            wi_raw["TechSupport"] = 1 if wi_tech == "Yes" else 0

            wi_proba, wi_X = _predict(wi_raw)
            wi_proba = wi_proba[0]
            baseline_proba = st.session_state["baseline_proba"]
            delta = wi_proba - baseline_proba

            st.markdown("---")
            wi_r1, wi_r2 = st.columns([1, 2])

            with wi_r1:
                st.metric("Baseline probability", f"{baseline_proba:.1%}")
                st.metric("Simulated probability", f"{wi_proba:.1%}", delta=f"{delta:+.1%}")
                wi_label, wi_kind = risk_band(wi_proba, FINAL_THRESHOLD)
                getattr(st, wi_kind)(f"Simulated risk: {wi_label}")

                st.markdown("**Top simulated risk factors**")
                wi_contrib = _shap_explanation(wi_X)
                wi_raw_row = wi_raw.iloc[0].to_dict()
                wi_reasons, _ = get_reasons_and_actions(wi_contrib, raw_row=wi_raw_row, n_top=3)
                for r in wi_reasons:
                    st.markdown(r)

            with wi_r2:
                st.markdown("**Simulated SHAP explanation**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                colors2 = ["#c0392b" if v > 0 else "#2980b9" for v in wi_contrib["shap_value"]]
                ax2.barh(wi_contrib["feature"], wi_contrib["shap_value"], color=colors2)
                ax2.set_xlabel("SHAP value (impact on churn probability)")
                ax2.invert_yaxis()
                st.pyplot(fig2)
                plt.close(fig2)

# ========================================================================
# MODE 2: Batch ranking
# ========================================================================
else:
    st.markdown("### Rank customers by churn risk")
    st.caption(
        "Upload a CSV with the same columns as the training data (one row "
        "per customer), or use the bundled sample test set to see how it "
        "works."
    )

    use_sample = st.checkbox("Use bundled sample test set", value=True)
    uploaded = None if use_sample else st.file_uploader("Upload customer CSV", type="csv")

    source_df = None
    if use_sample:
        try:
            source_df = pd.read_csv(f"{ARTIFACTS_DIR}/sample_test_set.csv")
        except FileNotFoundError:
            st.warning("No bundled sample file found -- upload a CSV instead.")
    elif uploaded is not None:
        source_df = pd.read_csv(uploaded)

    if source_df is not None:
        raw = source_df.copy()
        raw = encode_raw_inputs(raw)

        proba, _ = _predict(raw)

        ranked = source_df.copy()
        ranked["churn_probability"] = proba
        ranked = ranked.sort_values("churn_probability", ascending=False).reset_index(drop=True)
        ranked.insert(0, "priority_rank", range(1, len(ranked) + 1))

        n_capacity = st.slider(
            "Retention team capacity this month (top N customers)",
            min_value=10, max_value=min(500, len(ranked)), value=min(100, len(ranked)),
        )

        st.markdown(
            f"Showing the **top {n_capacity}** highest-risk customers out of "
            f"{len(ranked)} total. This is the recommended way to use the "
            f"model's output -- work down the ranked list to match actual "
            f"capacity, rather than a fixed probability cutoff."
        )

        # Light business-value framing: risk-band breakdown for the selected top N
        top_n = ranked.head(n_capacity)
        high = (top_n["churn_probability"] >= FINAL_THRESHOLD).sum()
        moderate = (
            (top_n["churn_probability"] >= FINAL_THRESHOLD * 0.6) &
            (top_n["churn_probability"] < FINAL_THRESHOLD)
        ).sum()
        low = (top_n["churn_probability"] < FINAL_THRESHOLD * 0.6).sum()
        st.markdown(
            f"Risk breakdown in top {n_capacity}: "
            f"🔴 **{high}** High | 🟡 **{moderate}** Moderate | 🟢 **{low}** Low"
        )

        display_cols = ["priority_rank", "churn_probability", "tenure",
                         "Contract", "MonthlyCharges"]
        display_cols = [c for c in display_cols if c in ranked.columns]
        st.dataframe(ranked[display_cols].head(n_capacity), use_container_width=True)

        csv = ranked.head(n_capacity).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download ranked list (CSV)", data=csv,
            file_name="churn_priority_list.csv", mime="text/csv",
        )

st.markdown("---")
st.caption(
    f"Model: XGBoost, test ROC-AUC {metadata['test_roc_auc']:.3f}, "
    f"F1 {metadata['test_f1_at_default']:.3f} at default threshold. "
    "See project README for full methodology, threshold reasoning, and limitations."
)