"""LLM glue code: prompt building and LLM call wrapper.

This module intentionally keeps the LLM call abstracted: by default it expects
an HTTP-based API endpoint configured with env vars (LLM_API_URL, LLM_API_KEY).

IMPORTANT: The LLM must never be used for decision-making. It receives the
decision, risk score and explanations as context and returns human-friendly text.
"""
from __future__ import annotations

import os
import requests
from typing import Dict, Any, List

LLM_API_URL = os.environ.get("LLM_API_URL")
LLM_API_KEY = os.environ.get("LLM_API_KEY")


def _pretty_feature_name(name: str) -> str:
    return name.replace("_", " ")


def build_explanation_prompt(decision: str, risk_score: float, shap_explanation: Dict[str, float], rag_contexts: List[Dict[str, Any]], sentiment: str) -> str:
    """Constructs a clear, constrained prompt for the LLM.

    The prompt enforces that the LLM NOT change the decision, and only produce
    an explanation, advisory, and (conditionally) a counter-offer suggestion.
    """
    lines = []
    lines.append("You are an expert financial advisor assisting with loan decisions.")
    lines.append("DO NOT change or question the machine decision. Use the inputs only to explain and advise.")
    lines.append("")
    lines.append(f"Decision: {decision}")
    lines.append(f"Risk score (higher is riskier): {risk_score:.3f}")
    lines.append("")
    lines.append("Top feature contributions (SHAP). Positive values increase risk):")
    for feat, val in sorted(shap_explanation.items(), key=lambda kv: -abs(kv[1]))[:10]:
        lines.append(f"- {feat}: {val:+.4f}")
    lines.append("")
    if rag_contexts:
        lines.append("Regulatory / policy supporting snippets (RAG):")
        for r in rag_contexts:
            payload = r.get("payload") or r
            text = payload.get("text") if isinstance(payload, dict) else str(payload)
            lines.append(f"- {text[:400]}")
        lines.append("")

    lines.append(f"User sentiment: {sentiment}")
    lines.append("")
    lines.append("Output format (JSON): { explanation: str, advisory: str, counter_offer: optional str or null, grounded_references: list }")
    lines.append("Keep explanations concise (max 250 words). When suggesting counter-offers, only propose them if the risk is borderline (0.4 <= risk <= 0.6). If risk is high (>0.7), explicitly state no counter-offer is recommended.")

    prompt = "\n".join(lines)
    return prompt


def build_rule_based_response(
    decision: str,
    risk_score: float,
    shap_explanation: Dict[str, float],
    sentiment: str,
    sample: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Fallback explanation/advisory bundle when an external LLM is unavailable."""
    ranked_features = sorted(shap_explanation.items(), key=lambda kv: -abs(kv[1]))
    risk_drivers = [_pretty_feature_name(name) for name, value in ranked_features if value > 0][:3]
    supportive_drivers = [_pretty_feature_name(name) for name, value in ranked_features if value < 0][:2]

    tone_prefix = {
        "frustrated": "I understand this outcome may be frustrating. ",
        "positive": "Here is a concise summary. ",
        "neutral": "",
    }.get(sentiment, "")

    if decision == "Rejected":
        explanation = (
            f"{tone_prefix}The application was rejected with risk score {risk_score:.2f}. "
            f"The biggest risk drivers were: {', '.join(risk_drivers) if risk_drivers else 'the evaluated financial features'}."
        )
    else:
        explanation = (
            f"{tone_prefix}The application was approved with risk score {risk_score:.2f}. "
            f"Helpful factors included: {', '.join(supportive_drivers) if supportive_drivers else 'the evaluated financial profile'}."
        )

    advice = []
    if "dti" in shap_explanation and shap_explanation["dti"] > 0:
        advice.append("Lower the debt-to-income ratio by reducing existing debt or requesting a smaller amount.")
    if "existing_loans" in shap_explanation and shap_explanation["existing_loans"] > 0:
        advice.append("Reducing or consolidating current loans would improve the profile.")
    if "credit_score" in shap_explanation and shap_explanation["credit_score"] > 0:
        advice.append("Improving the credit score would reduce future risk.")
    if "income" in shap_explanation and shap_explanation["income"] > 0:
        advice.append("Higher documented income or a co-applicant could strengthen affordability.")
    if not advice:
        advice.append("Maintain the current profile and avoid taking on additional debt before the next application.")

    sample = sample or {}
    counter_offer = None
    if 0.4 <= risk_score <= 0.6:
        offers = []
        dti = float(sample.get("dti", 0.0) or 0.0)
        existing_loans = float(sample.get("existing_loans", 0.0) or 0.0)
        credit_score = float(sample.get("credit_score", 0.0) or 0.0)
        loan_amount = float(sample.get("loan_amount", 0.0) or 0.0)
        tenure_months = int(sample.get("tenure_months", 0) or 0)

        if shap_explanation.get("dti", 0.0) > 0:
            target_dti = min(dti, 0.30) if dti else 0.30
            reduction_pct = max(10, min(30, int(round(max(dti - target_dti, 0.05) * 100))))
            exact_amount = None
            exact_tenure = None
            if loan_amount > 0:
                exact_amount = int(round(loan_amount * (1 - reduction_pct / 100.0)))
            if tenure_months > 0:
                exact_tenure = tenure_months + 12

            if exact_amount is not None and exact_tenure is not None:
                offers.append(
                    f"Counter-offer: sanction {exact_amount} over {exact_tenure} months instead of {int(loan_amount)} over {tenure_months} months so DTI moves closer to {target_dti:.2f}."
                )
            elif exact_amount is not None:
                offers.append(
                    f"Counter-offer: sanction {exact_amount}, which is about {reduction_pct}% below the requested amount, so DTI moves closer to {target_dti:.2f}."
                )
            elif exact_tenure is not None:
                offers.append(
                    f"Counter-offer: extend the tenure from {tenure_months} months to {exact_tenure} months so DTI moves closer to {target_dti:.2f}."
                )
            else:
                offers.append(
                    f"Counter-offer: reduce the requested loan amount by about {reduction_pct}% or extend tenure by 12 to 24 months so DTI moves closer to {target_dti:.2f}."
                )
        if shap_explanation.get("existing_loans", 0.0) > 0 and existing_loans > 0:
            target_loans = max(0, int(existing_loans) - 1)
            offers.append(
                f"Alternative: proceed after reducing existing loans from {int(existing_loans)} to {target_loans}, then reassess for standard pricing."
            )
        if shap_explanation.get("credit_score", 0.0) > 0:
            target_score = max(credit_score + 20, 680)
            offers.append(
                f"Alternative: offer a secured or co-applicant-backed loan now, or reapply after improving the credit score toward {int(target_score)}."
            )
        counter_offer = offers[0] if offers else "Counter-offer: sanction a smaller amount, about 10% to 15% lower, with tighter repayment terms."

    return {
        "explanation": explanation,
        "advisory": " ".join(advice[:2]),
        "counter_offer": counter_offer,
        "grounded_references": [],
    }


def call_llm(prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
    """Simple HTTP wrapper for LLMs. The environment must set LLM_API_URL and LLM_API_KEY.

    The function returns parsed JSON if the LLM returns JSON, else returns raw text under 'text'.
    """
    if not LLM_API_URL:
        raise RuntimeError("LLM_API_URL not set in environment")

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {"prompt": prompt, "max_tokens": max_tokens}
    resp = requests.post(LLM_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    text = resp.text
    try:
        return resp.json()
    except Exception:
        return {"text": text}
