"""Compliance report generation using Google Gemini.

Consumes:
- Extracted document data
- Applicant form declarations
- Mismatch flags (from comparator)
- RBI guideline snippets (from existing RAG pipeline)

Produces a structured compliance report dict.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# GEMINI_API_KEY must be set in the environment.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def _get_model():
    """Lazily configure and return a Gemini GenerativeModel instance."""
    import google.generativeai as genai  # noqa: PLC0415

    if not _GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Export it before starting the server: export GEMINI_API_KEY=<key>"
        )
    genai.configure(api_key=_GEMINI_API_KEY)
    return genai.GenerativeModel(_GEMINI_MODEL)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_compliance_prompt(
    extracted: Dict[str, Any],
    form_data: Dict[str, Any],
    mismatches: Dict[str, Any],
    rag_contexts: List[Dict[str, Any]],
    extraction_assessment: Dict[str, Any] | None = None,
) -> str:
    """Construct the LLM prompt for the compliance report."""
    rag_text = "\n".join(
        f"- {hit['payload']['text']}"
        for hit in rag_contexts
        if isinstance(hit.get("payload"), dict) and hit["payload"].get("text")
    ) or "No specific RBI guidelines retrieved for this query."

    extraction_text = json.dumps(extraction_assessment or {}, indent=2)

    return f"""You are an RBI compliance officer reviewing a retail loan application.

## Extracted Document Data (from uploaded PDFs)
{json.dumps(extracted, indent=2)}

## Applicant Declared Form Data
{json.dumps(form_data, indent=2)}

## Detected Mismatches
{json.dumps(mismatches, indent=2)}

## Extraction Quality Assessment
{extraction_text}

## Relevant RBI Guidelines (retrieved from internal knowledge base)
{rag_text}

## Instructions
Analyse the data above and produce a compliance report in the exact JSON schema below.
Do NOT include any text outside the JSON block.

{{
  "compliance_status": "Compliant" | "Not Compliant" | "Manual Review Required",
  "reasons": ["<specific reason referencing guideline or data>", ...],
  "verification": {{
    "income":       {{"declared": <number|null>, "extracted": <number|null>, "status": "Match" | "Mismatch" | "Unavailable"}},
    "loans":        {{"declared": <number|null>, "extracted": <number|null>, "status": "Match" | "Mismatch" | "Unavailable"}},
    "dti":          {{"value": <number|null>,    "status": "Within Limit" | "Exceeds Limit" | "Unavailable"}},
    "credit_score": {{"value": <number|null>,    "status": "Acceptable" | "Below Threshold" | "Unavailable"}}
  }},
  "recommendations": ["<actionable step for the applicant>", ...],
  "counter_offer": null | {{
    "description": "<brief description of alternate offer>",
    "conditions":  ["<condition 1>", ...]
  }},
  "manual_review_required": true | false,
  "warnings": ["<warning about extraction quality or missing data>", ...]
}}

Rules:
- Base compliance determination strictly on the RBI guidelines provided above.
- Do not invent guidelines not present in the retrieved context.
- If extraction_quality reports missing critical fields or review_required=true, set compliance_status to "Manual Review Required".
- Include a counter_offer only when the application is borderline (minor issues fixable within 6 months).
- Keep each reason and recommendation to one concise sentence."""


# ---------------------------------------------------------------------------
# Compliance report generator
# ---------------------------------------------------------------------------

def generate_compliance_report(
    extracted: Dict[str, Any],
    form_data: Dict[str, Any],
    mismatches: Dict[str, Any],
    rag_contexts: List[Dict[str, Any]],
    extraction_assessment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Generate an RBI compliance report.

    Calls Gemini when GEMINI_API_KEY is available; falls back to rule-based
    logic otherwise so the endpoint always returns a structured response.
    """
    prompt = build_compliance_prompt(extracted, form_data, mismatches, rag_contexts, extraction_assessment)

    if _GEMINI_API_KEY:
        try:
            model = _get_model()
            response = model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown code fences if Gemini wraps the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            return json.loads(raw.strip())
        except Exception as exc:
            # Log and fall through to rule-based fallback
            import logging
            logging.getLogger(__name__).warning("Gemini call failed: %s", exc)

    return _rule_based_compliance(extracted, form_data, mismatches, extraction_assessment)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_compliance(
    extracted: Dict[str, Any],
    form_data: Dict[str, Any],
    mismatches: Dict[str, Any],
    extraction_assessment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Deterministic compliance report used when Gemini is unavailable."""
    reasons: list[str] = []
    recommendations: list[str] = []
    warnings = list((extraction_assessment or {}).get("warnings", []))

    mismatch_map = mismatches.get("mismatches", mismatches)  # tolerate both shapes
    manual_review_required = bool((extraction_assessment or {}).get("review_required"))

    if manual_review_required:
        missing = ", ".join((extraction_assessment or {}).get("missing_critical_fields", []))
        reasons.append(
            "Critical document fields could not be extracted reliably"
            + (f": {missing}." if missing else ".")
        )
        recommendations.append(
            "Request manual review or upload a clearer supported document before making a compliance decision."
        )

    if mismatch_map.get("income_mismatch"):
        diff = mismatch_map["income_mismatch"]["difference_pct"]
        reasons.append(
            f"Income discrepancy of {diff}% detected between declared "
            "and document-extracted values."
        )
        recommendations.append(
            "Provide authentic payslips or bank statements that match the declared income."
        )

    if mismatch_map.get("hidden_loans"):
        count = mismatch_map["hidden_loans"]["undisclosed_count"]
        reasons.append(
            f"{count} undisclosed active loan(s) found in the credit report."
        )
        recommendations.append(
            "Declare all active credit facilities to ensure accurate DTI computation."
        )

    if mismatch_map.get("high_dti"):
        dti = mismatch_map["high_dti"]["dti"]
        reasons.append(
            f"DTI ratio of {dti:.1%} exceeds the RBI recommended ceiling of 50%."
        )
        recommendations.append(
            "Prepay or close existing EMIs to bring DTI below 50% before reapplying."
        )

    if mismatch_map.get("low_credit_score"):
        score = mismatch_map["low_credit_score"]["score"]
        reasons.append(
            f"Credit score of {score} is below the minimum acceptable threshold of 650."
        )
        recommendations.append(
            "Improve credit score by clearing overdue payments and reducing credit utilisation."
        )

    is_compliant = len(reasons) == 0 and not manual_review_required

    # Build verification block
    income_status = (
        "Mismatch" if mismatch_map.get("income_mismatch")
        else ("Unavailable" if extracted.get("income") is None else "Match")
    )
    loans_status = (
        "Mismatch" if mismatch_map.get("hidden_loans")
        else ("Unavailable" if extracted.get("active_loans") is None else "Match")
    )
    dti_status = (
        "Exceeds Limit" if mismatch_map.get("high_dti")
        else ("Unavailable" if extracted.get("dti") is None else "Within Limit")
    )
    score_val = extracted.get("credit_score")
    score_status = (
        "Unavailable" if score_val is None
        else ("Below Threshold" if score_val < 650 else "Acceptable")
    )

    # Counter-offer: only when exactly one fixable issue
    counter_offer = None
    if not is_compliant and len(reasons) == 1 and mismatch_map.get("high_dti"):
        counter_offer = {
            "description": "Consider a lower loan amount or extended tenure to reduce EMI burden.",
            "conditions": [
                "Bring DTI below 50% by reducing outstanding EMIs",
                "Maintain a credit score above 650",
            ],
        }

    return {
        "compliance_status": (
            "Manual Review Required"
            if manual_review_required
            else ("Compliant" if is_compliant else "Not Compliant")
        ),
        "reasons": reasons,
        "verification": {
            "income": {
                "declared": form_data.get("declared_income"),
                "extracted": extracted.get("income"),
                "status": income_status,
            },
            "loans": {
                "declared": form_data.get("declared_loans"),
                "extracted": extracted.get("active_loans"),
                "status": loans_status,
            },
            "dti": {
                "value": extracted.get("dti"),
                "status": dti_status,
            },
            "credit_score": {
                "value": score_val,
                "status": score_status,
            },
        },
        "recommendations": recommendations or ["Application meets all basic compliance requirements."],
        "counter_offer": counter_offer,
        "manual_review_required": manual_review_required,
        "warnings": warnings,
    }
