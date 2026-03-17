from __future__ import annotations

from datetime import datetime
import os
import re

import altair as alt
import pandas as pd
import requests
import streamlit as st



def normalize_service_url(value: str) -> str:
    return value if value.startswith(("http://", "https://")) else f"http://{value}"


API_BASE = normalize_service_url(os.getenv("API_BASE", "http://localhost:8000"))

st.set_page_config(page_title="XAI Finance Platform", layout="wide")
st.title("Explainable AI Financial Decision Platform")

page = st.sidebar.radio(
    "Pages",
    [
        "Model comparison",
        "Prediction input",
        "Explainability visualization",
        "Fairness metrics",
        "Monitoring metrics",
        "Audit logs",
        "Explanation feedback rating",
    ],
)


def show_backend_help(error: Exception) -> None:
    st.error("The FastAPI backend is not reachable on `http://localhost:8000`.")
    st.caption(str(error))
    st.info(
        "Start the API in a separate terminal with `python -m uvicorn backend_services.api:app --reload --app-dir .` "
        "from the project root. If it crashes during import, recreate the environment with Python 3.11 or 3.12."
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
        response = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        show_backend_help(exc)
        st.stop()


def prediction_label(prediction: int) -> str:
    return "Denied" if prediction == 1 else "Accepted"


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
                "Sex EO Diff": round(sex_fairness.get("equal_opportunity_difference", 0.0), 3),
                "Age DP Diff": round(age_fairness.get("demographic_parity_difference", 0.0), 3),
                "Age EO Diff": round(age_fairness.get("equal_opportunity_difference", 0.0), 3),
            }
        )
    return pd.DataFrame(rows)


def fairness_status(value: float) -> str:
    if value <= 0.05:
        return "Good"
    if value <= 0.10:
        return "Watch"
    return "Needs review"


def format_fairness_metrics(rows: list[dict]) -> pd.DataFrame:
    formatted_rows = []
    for row in rows:
        fairness = row.get("fairness", {})
        sex = fairness.get("Sex", {})
        age = fairness.get("age_group", {})
        worst_gap = max(
            float(sex.get("demographic_parity_difference", 0.0)),
            float(sex.get("equal_opportunity_difference", 0.0)),
            float(age.get("demographic_parity_difference", 0.0)),
            float(age.get("equal_opportunity_difference", 0.0)),
        )
        formatted_rows.append(
            {
                "Model": row.get("model_name", "").replace("_", " ").title(),
                "Version": row.get("model_version", ""),
                "Sex Selection Gap": round(float(sex.get("demographic_parity_difference", 0.0)), 3),
                "Sex Opportunity Gap": round(float(sex.get("equal_opportunity_difference", 0.0)), 3),
                "Age Selection Gap": round(float(age.get("demographic_parity_difference", 0.0)), 3),
                "Age Opportunity Gap": round(float(age.get("equal_opportunity_difference", 0.0)), 3),
                "Overall Status": fairness_status(worst_gap),
            }
        )
    return pd.DataFrame(formatted_rows)


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


def format_audit_logs(audit_rows: list[dict], feedback_rows: list[dict]) -> pd.DataFrame:
    feedback_map = {}
    for row in feedback_rows:
        request_id = row.get("request_id")
        rating = row.get("rating")
        if request_id:
            feedback_map[request_id] = rating

    formatted_rows = []
    for row in audit_rows:
        rating = feedback_map.get(row.get("request_id"))
        formatted_rows.append(
            {
                "Feedback": feedback_stars(rating),
                "Request ID": row.get("request_id", ""),
                "Model Version": row.get("model_version", ""),
                "Audit Hash": row.get("audit_hash", ""),
                "Timestamp": format_timestamp(row.get("timestamp", "")),
            }
        )
    return pd.DataFrame(formatted_rows)


def format_feedback_history(feedback_rows: list[dict]) -> pd.DataFrame:
    formatted_rows = []
    for row in feedback_rows:
        formatted_rows.append(
            {
                "Request ID": row.get("request_id", ""),
                "Rating": row.get("rating", ""),
                "Comment": row.get("comment", ""),
                "Submitted At": format_timestamp(row.get("created_at", "")),
            }
        )
    return pd.DataFrame(formatted_rows)


def feedback_stars(rating) -> str:
    if rating in (None, ""):
        return ""
    try:
        rating_value = int(rating)
    except (TypeError, ValueError):
        return ""
    rating_value = max(1, min(5, rating_value))
    return "★" * rating_value


def stability_label(score) -> str:
    if score is None:
        return "Not available"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "Not available"
    if value <= 0.33:
        return "Low agreement"
    if value <= 0.66:
        return "Moderate agreement"
    return "Strong agreement"


def humanize_feature_label(label: str) -> str:
    if not label:
        return ""
    cleaned = label.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("has_", "").replace("_", " ")
    cleaned = cleaned.replace("radio/tv", "radio or tv")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    operators = ["<=", ">=", "<", ">", "="]
    for operator in operators:
        if operator in cleaned:
            left, right = cleaned.split(operator, 1)
            return f"{left.strip().title()} {operator} {right.strip()}"
    if " " in cleaned:
        head, tail = cleaned.rsplit(" ", 1)
        if tail.lower() in {"male", "female", "own", "rent", "free", "little", "moderate", "rich", "unknown"}:
            return f"{head.title()}: {tail.title()}"
    return cleaned.title()


def prepare_explanation_frame(items: list[dict], sort_ascending: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(items).copy()
    if frame.empty:
        return frame
    frame["feature"] = frame["feature"].map(humanize_feature_label)
    frame["importance"] = frame["importance"].astype(float)
    return frame.sort_values("importance", ascending=sort_ascending)


def render_horizontal_explanation_chart(title: str, items: list[dict]) -> None:
    frame = prepare_explanation_frame(items)
    st.write(title)
    if frame.empty:
        st.info("No explanation data available.")
        return
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("feature:N", sort=None, title="Feature", axis=alt.Axis(labelAngle=0, labelLimit=220)),
            y=alt.Y("importance:Q", title="Impact"),
            color=alt.condition(
                alt.datum.importance >= 0,
                alt.value("#7bc8f6"),
                alt.value("#ff8f70"),
            ),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)


def render_rule_based_explanation(rule_data: dict) -> None:
    st.write("Rule-based explanation")
    status = rule_data.get("status")
    if status == "ready":
        decision = prediction_label(int(rule_data.get("prediction", 0)))
        st.caption(f"RIPPER rule engine decision: {decision} | {rule_data.get('alignment_with_model', '')}")
        matched_rule = rule_data.get("matched_rule", "")
        matched_rule_summary = rule_data.get("matched_rule_summary") or []
        if matched_rule_summary:
            summary_text = (
                f"This case was assessed as {decision.lower()} by the rule engine because "
                + ", ".join(matched_rule_summary[:-1])
                + (f", and {matched_rule_summary[-1]}." if len(matched_rule_summary) > 1 else f"{matched_rule_summary[0]}.")
            )
            st.write(summary_text[0].upper() + summary_text[1:])
        if matched_rule:
            st.write("Plain-language rule summary")
            st.success(matched_rule)
        if matched_rule_summary:
            st.write("Main reasons identified by the rule engine")
            st.markdown("\n".join(f"- {item.capitalize()}" for item in matched_rule_summary))
        case_summary = rule_data.get("case_summary") or []
        if case_summary:
            with st.expander("View case conditions used by the rule engine"):
                st.markdown("\n".join(f"- {item}" for item in case_summary))
        learned_rules = rule_data.get("learned_rules", "")
        if learned_rules:
            with st.expander("Technical rule details"):
                st.code(learned_rules, language="text")
    elif status == "unavailable":
        st.info(rule_data.get("message", "Rule-based explanation is unavailable in this environment."))
    else:
        st.warning(rule_data.get("message", "Rule-based explanation could not be generated."))


def render_counterfactual_result(data: dict) -> None:
    st.write("Optional counterfactual explanation")
    status = data.get("status")
    if status in {"not_requested", ""}:
        st.caption("Generate this only when you want an advanced what-could-change explanation from the separate Alibi service.")
        return
    if status in {"queued", "processing"}:
        st.info("Counterfactual generation is running in the separate Alibi service. Refresh in a few seconds.")
        return
    if status == "error":
        st.error(data.get("error_message", "Counterfactual generation failed."))
        return
    result = data.get("result", {})
    result_status = result.get("status")
    if result_status == "ready":
        st.success(result.get("message", "Counterfactual explanation is ready."))
        suggestions = result.get("suggested_changes", [])
        if suggestions:
            frame = pd.DataFrame(suggestions).copy()
            frame["feature"] = frame["feature"].map(humanize_feature_label)
            frame["direction"] = frame["direction"].str.title()
            frame["change_magnitude"] = frame["change_magnitude"].round(3)
            frame.columns = ["Feature", "Suggested Change", "Magnitude"]
            st.dataframe(frame, use_container_width=True, hide_index=True)
    elif result_status == "not_found":
        st.info(result.get("message", "No counterfactual was found."))
    elif result_status == "unavailable":
        st.warning(result.get("message", "Counterfactual service is unavailable."))
    else:
        st.warning(result.get("message", "Counterfactual result is unavailable."))


if page == "Model comparison":
    st.subheader("Top registered models")
    data = api_get("/models/comparison")
    st.dataframe(format_model_comparison(data), use_container_width=True, hide_index=True)
    st.caption(
        "Legend:  \nAccuracy = overall correctness  \nPrecision = how often predicted denials are truly risky"
        "  \nRecall = how many risky cases were caught  \nF1 = balance between precision and recall"
        "  \nROC-AUC = overall ranking quality  \nDP Diff = demographic parity difference"
        "  \nEO Diff = equal opportunity difference (Lower fairness difference values are generally better)."
    )

elif page == "Prediction input":
    st.subheader("Submit an applicant profile")
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        sex = st.selectbox("Sex", ["male", "female"])
        job = st.selectbox("Job", [0, 1, 2, 3], index=2)
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving_accounts = st.selectbox("Saving accounts", ["unknown", "little", "moderate", "rich", "quite rich"])
        checking_account = st.selectbox("Checking account", ["unknown", "little", "moderate", "rich"])
        credit_amount = st.number_input("Credit amount", min_value=100.0, value=2500.0)
        duration = st.number_input("Duration (months)", min_value=1, value=18)
        purpose = st.selectbox(
            "Purpose",
            ["radio/tv", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"],
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "age": age,
            "sex": sex,
            "job": job,
            "housing": housing,
            "saving_accounts": saving_accounts,
            "checking_account": checking_account,
            "credit_amount": credit_amount,
            "duration": duration,
            "purpose": purpose,
        }
        result = api_post("/predict", payload)
        st.session_state["latest_request_id"] = result["request_id"]
        st.success(
            f"Decision: {prediction_label(result['prediction'])} | Risk probability: {result['probability']:.3f}"
        )
        st.caption(f"Request ID: {result['request_id']}")

elif page == "Explainability visualization":
    st.subheader("Async explanation status")
    request_id = st.text_input("Request ID", value=st.session_state.get("latest_request_id", ""))
    col1, col2, col3 = st.columns(3)
    if col1.button("Refresh explanation") and request_id:
        data = api_get(f"/explanations/{request_id}")
        if data["status"] == "pending":
            st.info("Explanation is still processing. Try again in a few seconds.")
        elif data["status"] == "error":
            st.error(data.get("eli5_summary", "Explanation generation failed."))
        else:
            st.success("Explanation is ready")
            render_horizontal_explanation_chart("SHAP global importance", data["shap_global"])
            render_horizontal_explanation_chart("SHAP local explanation", data["shap_local"])
            render_horizontal_explanation_chart("LIME local explanation", data["lime_local"])
            render_rule_based_explanation(data.get("rule_based_explanation", {}))
            st.metric("Stability score", data["stability_score"])
            st.caption(f"Interpretation: {stability_label(data['stability_score'])}")
            st.text(data["eli5_summary"][:3000])
            counterfactual_data = api_get(f"/counterfactuals/{request_id}")
            render_counterfactual_result(counterfactual_data)
    if col2.button("Regenerate explanation") and request_id:
        api_post(f"/explanations/{request_id}/regenerate", {})
        st.success("Explanation regeneration queued. Refresh in a few seconds.")
    if col3.button("Generate counterfactual") and request_id:
        api_post(f"/counterfactuals/{request_id}/request", {})
        st.success("Counterfactual request queued. Refresh in a few seconds.")
    st.caption("If an explanation was generated before a dependency fix, use Regenerate explanation for the same request ID.")

elif page == "Fairness metrics":
    st.subheader("Fairness pre-checks")
    rows = api_get("/fairness")
    frame = format_fairness_metrics(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(
        "How to read this: smaller gap values are better. "
        "Selection Gap compares approval-rate differences between groups. "
        "Opportunity Gap compares how differently the model treats qualified cases across groups."
    )
    st.caption(
        "Status guide: Good = gap up to 0.05, Watch = 0.05 to 0.10, Needs review = above 0.10."
    )

elif page == "Monitoring metrics":
    st.subheader("Latency and stability")
    rows = api_get("/monitoring")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        st.dataframe(frame)
        numeric_cols = [col for col in ["prediction_latency_ms", "explanation_time_ms", "explanation_stability"] if col in frame]
        st.line_chart(frame[numeric_cols].fillna(0))
    else:
        st.info("No monitoring events yet.")

elif page == "Audit logs":
    st.subheader("Immutable audit trail simulation")
    audit_rows = api_get("/audit-logs")
    feedback_rows = api_get("/feedback")
    st.dataframe(format_audit_logs(audit_rows, feedback_rows), use_container_width=True, hide_index=True)
    st.caption("Feedback stars reflect the original 1-to-5 user rating for that request.")

elif page == "Explanation feedback rating":
    st.subheader("Rate explanation quality")
    col1, col2 = st.columns([2, 3])
    request_id = col1.text_input("Request ID for feedback", value=st.session_state.get("latest_request_id", ""))
    rating = col2.slider("Rating", min_value=1, max_value=5, value=4)
    comment = st.text_area("Comment")
    if st.button("Submit feedback"):
        api_post("/feedback", {"request_id": request_id, "rating": rating, "comment": comment})
        st.success("Feedback stored")

    st.write("Feedback history")
    feedback_rows = api_get("/feedback")
    feedback_frame = format_feedback_history(feedback_rows)
    if not feedback_frame.empty:
        st.dataframe(feedback_frame, use_container_width=True, hide_index=True)
    else:
        st.info("No feedback submitted yet.")
