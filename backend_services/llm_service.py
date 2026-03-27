from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.config import GEMINI_API_KEY, GEMINI_MODEL, LOWER_BORDERLINE_THRESHOLD, UPPER_BORDERLINE_THRESHOLD


def simple_sentiment(text: str | None) -> str:
    if not text:
        return "neutral"
    normalized = text.lower()
    if any(token in normalized for token in ["angry", "frustrat", "upset", "annoyed", "unfair"]):
        return "frustrated"
    if any(token in normalized for token in ["thanks", "thank", "great", "good", "happy"]):
        return "positive"
    return "neutral"


def should_generate_advisory(risk_score: float, ask_explain: bool) -> bool:
    return ask_explain or LOWER_BORDERLINE_THRESHOLD <= risk_score <= UPPER_BORDERLINE_THRESHOLD


def build_rule_based_response(
    decision: str,
    risk_score: float,
    shap_explanation: list[dict[str, Any]],
    sentiment: str,
    sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = sample or {}
    ranked = sorted(shap_explanation, key=lambda item: abs(float(item["importance"])), reverse=True)
    risk_drivers = [humanize_feature(item["feature"]) for item in ranked if float(item["importance"]) > 0][:3]
    support_drivers = [humanize_feature(item["feature"]) for item in ranked if float(item["importance"]) < 0][:2]

    tone_prefix = {
        "frustrated": "I understand this outcome can feel frustrating. ",
        "positive": "Here is a concise summary. ",
        "neutral": "",
    }.get(sentiment, "")

    if decision == "Rejected":
        explanation = (
            f"{tone_prefix}The application was rejected with risk score {risk_score:.2f}. "
            f"The strongest risk drivers were {', '.join(risk_drivers) if risk_drivers else 'the evaluated financial profile'}."
        )
    else:
        explanation = (
            f"{tone_prefix}The application was approved with risk score {risk_score:.2f}. "
            f"The strongest supportive factors were {', '.join(support_drivers) if support_drivers else 'the evaluated financial profile'}."
        )

    advice: list[str] = []
    shap_map = {strip_feature_prefix(item["feature"]): float(item["importance"]) for item in shap_explanation}
    if shap_map.get("dti", 0.0) > 0:
        advice.append("Lower the debt-to-income ratio by reducing debt or requesting a smaller amount.")
    if shap_map.get("existing_loans", 0.0) > 0:
        advice.append("Reducing or consolidating current loans would improve the profile.")
    if shap_map.get("credit_score", 0.0) > 0:
        advice.append("Improving the credit score would reduce future rejection risk.")
    if shap_map.get("income", 0.0) > 0:
        advice.append("Higher documented income or a co-applicant could strengthen affordability.")
    if not advice:
        advice.append("Maintain the current profile and avoid taking on additional debt before the next application.")

    counter_offer = None
    if LOWER_BORDERLINE_THRESHOLD <= risk_score <= UPPER_BORDERLINE_THRESHOLD:
        loan_amount = float(sample.get("loan_amount") or 0)
        tenure_months = int(sample.get("tenure_months") or 0)
        dti = float(sample.get("dti") or 0)
        if shap_map.get("dti", 0.0) > 0 and loan_amount > 0 and tenure_months > 0:
            reduced_amount = int(round(loan_amount * 0.9))
            extended_tenure = tenure_months + 12
            target_dti = max(dti - 0.05, 0.25)
            counter_offer = (
                f"Counter-offer: consider sanctioning {reduced_amount} over {extended_tenure} months "
                f"so the estimated DTI moves closer to {target_dti:.2f}."
            )
        elif shap_map.get("credit_score", 0.0) > 0:
            counter_offer = "Counter-offer: consider a secured or co-applicant-backed loan with tighter limits."
        else:
            counter_offer = "Counter-offer: consider a smaller sanctioned amount with slightly longer tenure."

    return {
        "explanation_text": explanation,
        "advisory": " ".join(advice[:2]),
        "counter_offer": counter_offer,
        "reports": build_reports(
            decision=decision,
            risk_score=risk_score,
            shap_explanation=shap_explanation,
            sentiment=sentiment,
            explanation_text=explanation,
            advisory=" ".join(advice[:2]),
            counter_offer=counter_offer,
            sample=sample,
        ),
    }


def build_explanation_prompt(
    decision: str,
    risk_score: float,
    shap_explanation: list[dict[str, Any]],
    sentiment: str,
    rag_context: list[dict[str, Any]] | None = None,
) -> str:
    rag_context = rag_context or []
    lines = [
        "You are an assistant for an AI-based loan decision system.",
        "Do not alter the model decision. Only explain it, provide advice, and suggest a counter-offer only for borderline cases.",
        f"Decision: {decision}",
        f"Risk score: {risk_score:.4f}",
        f"User sentiment: {sentiment}",
        "Top SHAP feature impacts:",
    ]
    for item in shap_explanation[:5]:
        lines.append(f"- {humanize_feature(item['feature'])}: {float(item['importance']):+.4f}")
    if rag_context:
        lines.append("Grounded policy context:")
        for item in rag_context[:3]:
            lines.append(f"- {json.dumps(item)}")
    lines.append(
        "Return valid JSON with keys explanation, advisory, counter_offer, reports, grounded_references. "
        "The reports field must be a list of objects with keys title, audience, summary, bullets. "
        "Include sensible report sections such as decision summary, rejection reasons if applicable, counter-offer if applicable, and improvement plan. "
        "Keep the explanation under 220 words."
    )
    return "\n".join(lines)


def call_gemini(prompt: str) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "explanation": {"type": "STRING"},
            "advisory": {"type": "STRING"},
            "counter_offer": {"type": "STRING", "nullable": True},
            "reports": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "audience": {"type": "STRING"},
                        "summary": {"type": "STRING"},
                        "bullets": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                    },
                    "required": ["title", "audience", "summary", "bullets"],
                },
            },
            "grounded_references": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": ["explanation", "advisory", "reports", "grounded_references"],
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    text = (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )
    if not text:
        return {"raw_text": "", "parsed": {}}
    parsed = _parse_json_like_response(text)
    return {"raw_text": text, "parsed": parsed}


def build_customer_chat_prompt(
    message: str,
    history: list[dict[str, Any]] | None = None,
    applicant_context: dict[str, Any] | None = None,
) -> str:
    history = history or []
    applicant_context = applicant_context or {}

    lines = [
        "You are a customer-facing loan advisory assistant for an AI-based loan decision system.",
        "Be clear, polite, concise, and practical.",
        "Do not claim final approval authority and do not invent policy rules.",
        "If applicant context is provided, use it to personalize the reply.",
        "If the customer asks why a loan was rejected, explain likely drivers in plain language.",
        "If they ask what to improve, give actionable next steps.",
    ]
    if applicant_context:
        lines.append(f"Applicant context: {json.dumps(applicant_context)}")
    if history:
        lines.append("Conversation history:")
        for item in history[-10:]:
            role = item.get("role", "user")
            content = item.get("content", "")
            lines.append(f"{role.title()}: {content}")
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat_with_customer(
    message: str,
    history: list[dict[str, Any]] | None = None,
    applicant_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = build_customer_chat_prompt(message, history, applicant_context)
    try:
        response = call_gemini(prompt)
        reply = response.get("raw_text", "").strip()
        if not reply:
            raise RuntimeError("Empty response from Gemini")
        return {"reply": reply, "source": "gemini"}
    except Exception:
        context_note = ""
        if applicant_context:
            decision = applicant_context.get("decision")
            risk_score = applicant_context.get("risk_score")
            if decision and risk_score is not None:
                context_note = f" Based on the latest case, the decision is {decision} with risk score {risk_score:.2f}."
        return {
            "reply": (
                "I can help explain the loan outcome, discuss possible improvements, and suggest what details to review next."
                f"{context_note} Please tell me what you would like to understand."
            ),
            "source": "fallback",
        }


def strip_feature_prefix(feature: str) -> str:
    return feature.replace("num__", "").replace("cat__", "")


def humanize_feature(feature: str) -> str:
    return strip_feature_prefix(feature).replace("_", " ")


def _parse_json_like_response(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(item.strip() for item in fenced if item.strip())

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def build_reports(
    decision: str,
    risk_score: float,
    shap_explanation: list[dict[str, Any]],
    sentiment: str,
    explanation_text: str,
    advisory: str,
    counter_offer: str | None,
    sample: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sample = sample or {}
    ranked = sorted(shap_explanation, key=lambda item: abs(float(item["importance"])), reverse=True)
    risk_drivers = [humanize_feature(item["feature"]).title() for item in ranked if float(item["importance"]) > 0][:3]
    support_drivers = [humanize_feature(item["feature"]).title() for item in ranked if float(item["importance"]) < 0][:3]

    reports: list[dict[str, Any]] = [
        {
            "title": "Decision Summary Report",
            "audience": "User",
            "summary": f"Loan decision is {decision} with risk score {risk_score:.2f}.",
            "bullets": [
                f"Sentiment detected from user input: {sentiment}.",
                explanation_text,
                f"Primary advisory: {advisory}" if advisory else "No additional advisory generated.",
            ],
        }
    ]

    if decision == "Rejected":
        reports.append(
            {
                "title": "Rejection Reason Report",
                "audience": "User and Loan Officer",
                "summary": "This report explains why the application crossed the model's rejection threshold.",
                "bullets": [
                    *(f"{driver} increased risk materially for this application." for driver in risk_drivers),
                    "The combined SHAP impact pushed the application above the rejection threshold.",
                ],
            }
        )
    else:
        reports.append(
            {
                "title": "Approval Basis Report",
                "audience": "User and Loan Officer",
                "summary": "This report explains the strongest factors supporting approval.",
                "bullets": [
                    *(f"{driver} supported approval for this application." for driver in support_drivers),
                    "The overall feature profile remained within the model's acceptable risk range.",
                ],
            }
        )

    if counter_offer:
        reports.append(
            {
                "title": "Counter-Offer Report",
                "audience": "Loan Officer",
                "summary": "This report summarizes the recommended alternative offer for a borderline or improvable case.",
                "bullets": [
                    counter_offer,
                    f"Requested amount: {sample.get('loan_amount') or 'N/A'}.",
                    f"Requested tenure: {sample.get('tenure_months') or 'N/A'} months.",
                ],
            }
        )

    if advisory:
        reports.append(
            {
                "title": "Improvement Plan Report",
                "audience": "User",
                "summary": "This report lists the most practical actions that could improve future eligibility.",
                "bullets": [item.strip() for item in advisory.split(". ") if item.strip()],
            }
        )

    reports.append(
        {
            "title": "Model Explainability Report",
            "audience": "Technical Reviewer",
            "summary": "This report summarizes the main local SHAP drivers used to explain the decision.",
            "bullets": [
                *(f"{humanize_feature(item['feature']).title()}: SHAP impact {float(item['importance']):+.4f}" for item in ranked[:5]),
            ],
        }
    )

    return reports
