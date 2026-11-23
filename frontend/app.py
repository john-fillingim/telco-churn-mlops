"""Streamlit front-end for the Telco Churn API.

Scores one customer and runs a contract "what-if" (the decision part of data -> decisions).
API_URL comes from the environment (a Hugging Face Space variable); it defaults to the live API.
Run locally: streamlit run frontend/app.py
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_URL = os.getenv(
    "API_URL", "https://telco-churn-api-681878559627.us-central1.run.app"
).rstrip("/")
GITHUB_URL = "https://github.com/john-fillingim/telco-churn-mlops"
TIMEOUT_S = 30  # Cloud Run scales to zero, so the first call can take a while
RISK_COLORS = {"low": "#2e7d32", "medium": "#ef6c00", "high": "#c62828"}

# Used only if /model-info can't be reached, so the form still renders
FALLBACK_SCHEMA: dict[str, dict[str, Any]] = {
    "tenure": {"type": "int", "min": 0, "max": 72},
    "MonthlyCharges": {"type": "float", "min": 18.0, "max": 120.0},
    "Contract": {"type": "category", "values": ["Month-to-month", "One year", "Two year"]},
    "InternetService": {"type": "category", "values": ["DSL", "Fiber optic", "No"]},
    "PaymentMethod": {
        "type": "category",
        "values": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    },
    "TechSupport": {"type": "category", "values": ["No", "No internet service", "Yes"]},
    "OnlineSecurity": {"type": "category", "values": ["No", "No internet service", "Yes"]},
    "PaperlessBilling": {"type": "category", "values": ["No", "Yes"]},
}
DEFAULTS = {"tenure": 12, "MonthlyCharges": 70.0}
LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "TechSupport": "Tech support",
    "OnlineSecurity": "Online security",
    "PaperlessBilling": "Paperless billing",
}


class ApiError(Exception):
    """A user-friendly error from talking to the API."""


def call_api(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the API and turn network/HTTP problems into friendly ApiError messages."""
    try:
        r = requests.request(method, f"{API_URL}{path}", json=payload, timeout=TIMEOUT_S)
    except requests.Timeout as exc:
        raise ApiError("The model server took too long to respond. Please try again.") from exc
    except requests.ConnectionError as exc:
        raise ApiError("Couldn't reach the model server. Check your connection.") from exc
    if r.status_code == 422:
        raise ApiError("The API rejected these inputs as invalid.")
    if not r.ok:
        raise ApiError(f"The model server returned an error (HTTP {r.status_code}).")
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_model_info() -> dict | None:
    try:
        return call_api("GET", "/model-info")
    except ApiError as exc:
        logger.warning("Falling back to built-in schema: %s", exc)
        return None


def customer_form(schema: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    """Build inputs from the feature schema: sliders for numbers, selectboxes for categories."""
    values: dict[str, Any] = {}
    with st.form("customer"):
        cols = st.columns(2)
        for i, (name, spec) in enumerate(schema.items()):
            col = cols[i % 2]
            label = LABELS.get(name, name)
            if spec["type"] == "int":
                values[name] = col.slider(
                    label, int(spec["min"]), int(spec["max"]), DEFAULTS.get(name, int(spec["min"]))
                )
            elif spec["type"] == "float":
                values[name] = col.slider(
                    label, float(spec["min"]), float(spec["max"]),
                    float(DEFAULTS.get(name, spec["min"])), step=0.5,
                )
            else:
                values[name] = col.selectbox(label, spec["values"])
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    return submitted, values


def show_prediction(result: dict) -> None:
    prob, risk = result["churn_probability"], result["risk_level"]
    c1, c2 = st.columns([1, 1])
    c1.metric("Churn probability", f"{prob:.0%}")
    c2.markdown(
        f"<div style='margin-top:1.6rem'><span style='background:{RISK_COLORS[risk]};"
        f"color:white;padding:0.4rem 0.9rem;border-radius:1rem;font-weight:600'>"
        f"{risk.upper()} RISK</span>&nbsp; Prediction: <b>{result['prediction']}</b></div>",
        unsafe_allow_html=True,
    )
    st.progress(prob)


def show_what_if(customer: dict[str, Any], contracts: list[str]) -> None:
    """Re-score the same customer under each contract type."""
    st.subheader("What if they switched contracts?")
    rows = []
    for contract in contracts:
        res = call_api("POST", "/predict", {**customer, "Contract": contract})
        rows.append({"Contract": contract, "Churn probability": res["churn_probability"]})
    df = pd.DataFrame(rows).set_index("Contract")
    st.bar_chart(df, y="Churn probability", horizontal=True)

    current = df.loc[customer["Contract"], "Churn probability"]
    best = df["Churn probability"].idxmin()
    if best != customer["Contract"]:
        st.info(
            f"Moving this customer from **{customer['Contract']}** to **{best}** would lower "
            f"churn risk from **{current:.0%}** to **{df.loc[best, 'Churn probability']:.0%}**. "
            "That makes them a good candidate for a contract-upgrade retention offer."
        )
    else:
        st.success(f"**{best}** is already the lowest-risk contract for this customer.")


def sidebar(info: dict | None) -> None:
    st.sidebar.header("Model")
    if info:
        st.sidebar.write(f"**Type:** {info['model_type'].replace('_', ' ')}")
        st.sidebar.write(f"**Version:** `{info['model_version']}`")
        st.sidebar.write(f"**Test ROC-AUC:** {info['test_metrics']['roc_auc']:.3f}")
    else:
        st.sidebar.warning("Model info unavailable (server may be waking up).")
    st.sidebar.markdown(f"[API docs]({API_URL}/docs) · [GitHub repo]({GITHUB_URL})")
    st.sidebar.caption(
        "Intended use: prioritizing retention outreach. Not for denying service or "
        "setting prices."
    )


def main() -> None:
    st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")
    st.title("📉 Telco Churn Predictor")
    st.caption("Estimate a customer's churn risk and see which contract would help keep them.")

    with st.spinner("Connecting to the model (the first load can take ~20s)..."):
        info = get_model_info()
    sidebar(info)
    schema = info["features"] if info else FALLBACK_SCHEMA

    submitted, customer = customer_form(schema)
    if not submitted:
        return
    try:
        with st.spinner("Scoring customer..."):
            show_prediction(call_api("POST", "/predict", customer))
            show_what_if(customer, schema["Contract"]["values"])
    except ApiError as exc:
        st.error(str(exc))


main()
