from __future__ import annotations

from datetime import datetime
import os

import altair as alt
import json
import pandas as pd
import requests
import streamlit as st


def normalize_service_url(value: str) -> str:
    return value if value.startswith(("http://", "https://")) else f"http://{value}"


API_BASE = normalize_service_url(os.getenv("API_BASE", "http://localhost:8000"))

st.set_page_config(page_title="AI-Based Loan Decision System", layout="wide")
st.title("AI-Based Loan Decision System")

page = st.sidebar.radio(
    "Pages",
    [
        "Model comparison",
        "Loan assessment",
        "Explainability",
        "Customer Chat",
        "Fairness metrics",
        "Monitoring metrics",
        "Audit logs",
        "Feedback",
    ],
)


def show_backend_help(error: Exception) -> None:
    st.error("The FastAPI backend is not reachable on `http://localhost:8000`.")
    st.caption(str(error))
    st.info(
        "Start the API in a separate terminal with `python -m uvicorn backend_services.api:app --reload --app-dir .` "
        "from the project root."
    )


def api_get(path: str):
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        show_backend_help(exc)
        st.stop()


def api_post(path: str, payload: dict):
    try:
        response = requests.post(f"{API_BASE}{path}", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        show_backend_help(exc)
        st.stop()


def format_timestamp(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime("%d %b %Y, %I:%M:%S %p")
    except ValueError:
        return value


def humanize_feature_label(label: str) -> str:
    return label.replace("num__", "").replace("cat__", "").replace("_", " ").title()


def render_horizontal_explanation_chart(title: str, items: list[dict]) -> None:
    frame = pd.DataFrame(items).copy()
    st.write(title)
    if frame.empty:
        st.info("No explanation data available.")
        return
    frame["feature"] = frame["feature"].map(humanize_feature_label)
    frame["importance"] = frame["importance"].astype(float)
    frame = frame.sort_values("importance", ascending=True)
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Impact"),
            y=alt.Y("feature:N", sort=None, title="Feature"),
            color=alt.condition(alt.datum.importance >= 0, alt.value("#e76f51"), alt.value("#2a9d8f")),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_reports(reports: list[dict]) -> None:
    st.write("Decision Reports")
    if not reports:
        st.info("No report sections available.")
        return
    for report in reports:
        title = report.get("title", "Report")
        audience = report.get("audience", "General")
        summary = report.get("summary", "")
        bullets = report.get("bullets", [])
        with st.expander(f"{title} | Audience: {audience}", expanded=True):
            if summary:
                st.write(summary)
            for bullet in bullets:
                st.markdown(f"- {bullet}")


def format_model_comparison(data: list[dict]) -> pd.DataFrame:
    rows = []
    for record in data:
        fairness = record.get("metrics", {}).get("fairness", {})
        sex_fairness = fairness.get("Sex", {})
        age_fairness = fairness.get("age_group", {})
        rows.append(
            {
                "Model": record.get("name", "").replace("_", " ").title(),
                "Version": record.get("version", ""),
                "Accuracy": round(record.get("metrics", {}).get("accuracy", 0.0), 3),
                "Precision": round(record.get("metrics", {}).get("precision", 0.0), 3),
                "Recall": round(record.get("metrics", {}).get("recall", 0.0), 3),
                "F1": round(record.get("metrics", {}).get("f1_score", 0.0), 3),
                "ROC-AUC": round(record.get("metrics", {}).get("roc_auc", 0.0), 3),
                "Sex DP Diff": round(sex_fairness.get("demographic_parity_difference", 0.0), 3),
                "Age DP Diff": round(age_fairness.get("demographic_parity_difference", 0.0), 3),
            }
        )
    return pd.DataFrame(rows)


def format_request_option(row: dict) -> str:
    request_id = row.get("request_id", "")
    decision = row.get("decision") or "Unknown"
    risk_score = row.get("risk_score")
    created_at = format_timestamp(row.get("created_at", ""))
    risk_text = f"{float(risk_score):.3f}" if risk_score is not None else "N/A"
    return f"{request_id} | {decision} | Risk {risk_text} | {created_at}"


def parse_json_field(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


if page == "Model comparison":
    st.subheader("Registered model")
    data = api_get("/models/comparison")
    st.dataframe(format_model_comparison(data), use_container_width=True, hide_index=True)

elif page == "Loan assessment":
    st.subheader("Submit applicant profile")
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=32, step=1)
        income = st.number_input("Income", min_value=1000.0, value=50000.0, step=1000.0)
        credit_score = st.number_input("Credit score", min_value=300.0, max_value=850.0, value=690.0, step=1.0)
        dti = st.number_input("Debt-to-income ratio", min_value=0.0, max_value=2.0, value=0.32, step=0.01)
        employment_length = st.number_input("Employment length (years)", min_value=0.0, value=6.0, step=0.5)
        existing_loans = st.number_input("Existing loans", min_value=0.0, value=1.0, step=1.0)
        loan_amount = st.number_input("Requested loan amount", min_value=1000.0, value=300000.0, step=1000.0)
        tenure_months = st.number_input("Tenure (months)", min_value=1, value=36, step=1)
        ask_explain = st.checkbox("Request explanation/advisory", value=True)
        user_text = st.text_area("User message", value="Please explain the decision.")
        submitted = st.form_submit_button("Assess loan")

    if submitted:
        payload = {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "dti": dti,
            "employment_length": employment_length,
            "existing_loans": existing_loans,
            "loan_amount": loan_amount,
            "tenure_months": tenure_months,
            "ask_explain": ask_explain,
            "user_text": user_text,
        }
        result = api_post("/predict", payload)
        st.session_state["latest_payload"] = payload
        st.session_state["latest_request_id"] = result["request_id"]
        st.success(f"Decision: {result['decision']} | Risk score: {result['risk_score']:.3f}")
        st.caption(f"Request ID: {result['request_id']} | Explanation status: {result['explanation_status']}")

elif page == "Explainability":
    st.subheader("SHAP explanation and advisory")
    request_id = st.text_input("Request ID", value=st.session_state.get("latest_request_id", ""))
    col1, col2 = st.columns(2)
    if col1.button("Refresh async explanation") and request_id:
        data = api_get(f"/explanations/{request_id}")
        if data["status"] == "pending":
            st.info("Explanation is still processing. Refresh again in a few seconds.")
        elif data["status"] == "error":
            st.error(data.get("explanation_text", "Explanation generation failed."))
        else:
            st.success(f"{data['decision']} | Risk score: {data['risk_score']:.3f}")
            st.session_state["latest_explanation_context"] = {
                "request_id": request_id,
                "decision": data.get("decision"),
                "risk_score": data.get("risk_score"),
                "explanation_text": data.get("explanation_text"),
                "advisory": data.get("advisory"),
                "counter_offer": data.get("counter_offer"),
            }
            render_reports(data.get("reports", []))
            render_horizontal_explanation_chart("Global SHAP importance", data["shap_global"])
            render_horizontal_explanation_chart("Local SHAP explanation", data["shap_local"])
            st.write(f"Sentiment: {data['sentiment']}")
            st.write("Explanation")
            st.info(data["explanation_text"])
            st.write("Advisory")
            st.success(data["advisory"] or "No advisory generated.")
            if data.get("counter_offer"):
                st.write("Counter-offer")
                st.warning(data["counter_offer"])
    if col2.button("Run synchronous explanation"):
        payload = st.session_state.get("latest_payload")
        if not payload:
            st.info("Submit a loan assessment first.")
        else:
            data = api_post("/explain", payload)
            st.session_state["latest_request_id"] = data["request_id"]
            st.session_state["latest_explanation_context"] = {
                "request_id": data.get("request_id"),
                "decision": data.get("decision"),
                "risk_score": data.get("risk_score"),
                "explanation_text": data.get("explanation_text"),
                "advisory": data.get("advisory"),
                "counter_offer": data.get("counter_offer"),
            }
            st.success(f"{data['decision']} | Risk score: {data['risk_score']:.3f}")
            render_reports(data.get("reports", []))
            render_horizontal_explanation_chart("Global SHAP importance", data["shap_global"])
            render_horizontal_explanation_chart("Local SHAP explanation", data["shap_local"])
            st.info(data["explanation_text"])
            st.success(data["advisory"] or "No advisory generated.")
            if data.get("counter_offer"):
                st.warning(data["counter_offer"])

elif page == "Customer Chat":
    st.subheader("Customer chat assistant")
    st.caption("This page uses Gemini to chat with the customer in a plain-language advisory style.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    prediction_rows = api_get("/predictions")
    request_options = {format_request_option(row): row["request_id"] for row in prediction_rows}
    if request_options:
        selected_request_label = st.selectbox(
            "Select request ID for chat context",
            options=list(request_options.keys()),
            index=0,
        )
        selected_request_id = request_options[selected_request_label]
    else:
        selected_request_id = None

    latest_context = st.session_state.get("latest_explanation_context", {})
    latest_payload = st.session_state.get("latest_payload", {})
    if latest_payload or latest_context:
        with st.expander("Current case context", expanded=False):
            if latest_payload:
                st.write("Applicant input")
                st.json(latest_payload)
            if latest_context:
                st.write("Latest decision context")
                st.json(latest_context)

    for item in st.session_state["chat_history"]:
        with st.chat_message("assistant" if item["role"] == "assistant" else "user"):
            st.write(item["content"])

    prompt = st.chat_input("Ask the assistant about the loan outcome, next steps, or customer concerns")
    if prompt:
        applicant_context = {}
        if latest_payload:
            applicant_context.update(latest_payload)
        if latest_context:
            applicant_context.update(latest_context)

        with st.chat_message("user"):
            st.write(prompt)

        result = api_post(
            "/chat",
            {
                "message": prompt,
                "history": st.session_state["chat_history"],
                "applicant_context": applicant_context,
            },
        )
        st.session_state["chat_history"] = result["history"]
        with st.chat_message("assistant"):
            st.write(result["reply"])
            st.caption(f"Source: {result['source']}")

    col1, col2, col3 = st.columns(3)
    if col1.button("Use latest explanation in chat"):
        payload = st.session_state.get("latest_payload")
        if payload:
            data = api_post("/explain", payload)
            st.session_state["latest_request_id"] = data.get("request_id")
            st.session_state["latest_explanation_context"] = {
                "request_id": data.get("request_id"),
                "decision": data.get("decision"),
                "risk_score": data.get("risk_score"),
                "explanation_text": data.get("explanation_text"),
                "advisory": data.get("advisory"),
                "counter_offer": data.get("counter_offer"),
                "reports": data.get("reports", []),
            }
            st.success("Latest explanation and reports loaded into chat.")
        else:
            st.info("Submit a loan assessment first so the app has context to explain.")
    if col2.button("Load selected request into chat"):
        if selected_request_id:
            prediction_data = api_get(f"/predictions/{selected_request_id}")
            payload = prediction_data.get("input_payload", {})
            data = api_post("/explain", payload)
            st.session_state["latest_payload"] = payload
            st.session_state["latest_request_id"] = data.get("request_id")
            st.session_state["latest_explanation_context"] = {
                "request_id": data.get("request_id"),
                "decision": data.get("decision"),
                "risk_score": data.get("risk_score"),
                "explanation_text": data.get("explanation_text"),
                "advisory": data.get("advisory"),
                "counter_offer": data.get("counter_offer"),
                "reports": data.get("reports", []),
            }
            st.success("Selected request context and reports loaded into chat.")
        else:
            st.info("No request IDs are available yet.")
    if col3.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.success("Chat history cleared.")

elif page == "Fairness metrics":
    st.subheader("Fairness pre-checks")
    rows = api_get("/fairness")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

elif page == "Monitoring metrics":
    st.subheader("Latency")
    rows = api_get("/monitoring")
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("No monitoring data yet.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)

elif page == "Audit logs":
    st.subheader("Audit trail")
    rows = api_get("/audit-logs")
    frame = pd.DataFrame(rows)
    if frame.empty:
        st.info("No audit logs yet.")
    else:
        if "timestamp" in frame:
            frame["timestamp"] = frame["timestamp"].map(format_timestamp)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        selected_request_id = st.selectbox("View applicant input for request", options=frame["request_id"].tolist())
        selected_row = next((row for row in rows if row.get("request_id") == selected_request_id), None)
        if selected_row:
            payload = parse_json_field(selected_row.get("payload"))
            applicant_input = payload.get("input_payload", {})
            if applicant_input:
                st.write("Applicant input")
                st.json(applicant_input)
            explanation_payload = payload.get("explanation", {})
            if explanation_payload:
                st.write("Stored explanation context")
                st.json(
                    {
                        "decision": payload.get("decision"),
                        "risk_score": payload.get("risk_score"),
                        "reports": explanation_payload.get("reports", []),
                        "advisory": explanation_payload.get("advisory"),
                        "counter_offer": explanation_payload.get("counter_offer"),
                    }
                )
            if st.button("Load selected audit context into chat"):
                st.session_state["latest_payload"] = applicant_input
                st.session_state["latest_request_id"] = selected_request_id
                st.session_state["latest_explanation_context"] = {
                    "request_id": selected_request_id,
                    "decision": payload.get("decision"),
                    "risk_score": payload.get("risk_score"),
                    "explanation_text": explanation_payload.get("explanation_text"),
                    "advisory": explanation_payload.get("advisory"),
                    "counter_offer": explanation_payload.get("counter_offer"),
                    "reports": explanation_payload.get("reports", []),
                }
                st.success("Selected audit context loaded into chat.")

elif page == "Feedback":
    st.subheader("Feedback")
    request_id = st.text_input("Request ID for feedback", value=st.session_state.get("latest_request_id", ""))
    rating = st.slider("Rating", min_value=1, max_value=5, value=4)
    comment = st.text_area("Comment")
    if st.button("Submit feedback"):
        api_post("/feedback", {"request_id": request_id, "rating": rating, "comment": comment})
        st.success("Feedback stored.")
    rows = api_get("/feedback")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        if "created_at" in frame:
            frame["created_at"] = frame["created_at"].map(format_timestamp)
        st.dataframe(frame, use_container_width=True, hide_index=True)
