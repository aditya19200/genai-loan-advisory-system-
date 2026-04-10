"""PDF extraction module for bank statements and credit reports.

Capabilities:
- direct text extraction via pdfplumber
- optional OCR fallback for scanned PDFs when optional OCR dependencies exist
- regex parsing for known bank-statement and credit-report layouts
- optional LLM-assisted extraction to recover fields from unfamiliar layouts
- document-specific validation metadata and confidence scoring
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any, Dict

import pdfplumber


_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return concatenated text from all pages of a PDF."""
    return extract_text_bundle(file_bytes)["text"]


def extract_text_bundle(file_bytes: bytes) -> Dict[str, Any]:
    """Extract text and metadata, using OCR as a best-effort fallback."""
    parts: list[str] = []
    page_count = 0
    warnings: list[str] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)

    text = "\n".join(parts).strip()
    method = "pdf_text"
    ocr_used = False

    if not text:
        ocr_text, ocr_warning = _extract_text_with_ocr(file_bytes)
        if ocr_warning:
            warnings.append(ocr_warning)
        if ocr_text:
            text = ocr_text.strip()
            method = "ocr"
            ocr_used = True

    return {
        "text": text,
        "page_count": page_count,
        "char_count": len(text),
        "extraction_method": method,
        "ocr_used": ocr_used,
        "warnings": warnings,
    }


def _extract_text_with_ocr(file_bytes: bytes) -> tuple[str, str | None]:
    """Best-effort OCR fallback for scanned PDFs."""
    try:
        import pypdfium2 as pdfium  # noqa: PLC0415
        import pytesseract  # noqa: PLC0415
    except ImportError:
        return "", "OCR fallback unavailable because optional OCR dependencies are not installed."

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(file_bytes))
        pages_text: list[str] = []
        for page_index in range(len(pdf)):
            page = pdf.get_page(page_index)
            bitmap = page.render(scale=2)
            pil_image = bitmap.to_pil()
            page_text = pytesseract.image_to_string(pil_image)
            if page_text:
                pages_text.append(page_text)
            page.close()
        return "\n".join(pages_text), None
    except Exception as exc:
        return "", f"OCR fallback failed: {exc}"


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

_BANK_KEYWORDS = [
    "account statement", "bank statement", "transaction", "balance",
    "debit", "withdrawal", "deposit", "opening balance", "closing balance",
]
_CREDIT_KEYWORDS = [
    "credit score", "credit report", "credit bureau", "cibil score",
    "equifax", "experian", "credit rating", "active loans",
    "credit history", "credit summary", "summary of accounts",
    "installment", "revolving", "transunion",
]


def classify_document(text: str) -> str:
    """Return 'bank_statement' or 'credit_report' based on keyword heuristic."""
    lower = text.lower()
    bank_hits = sum(1 for kw in _BANK_KEYWORDS if kw in lower)
    credit_hits = sum(1 for kw in _CREDIT_KEYWORDS if kw in lower)
    return "credit_report" if credit_hits > bank_hits else "bank_statement"


# ---------------------------------------------------------------------------
# Bank statement parsing
# ---------------------------------------------------------------------------

_SALARY_KEYWORDS = [
    "salary", "sal ", "neft cr", "imps cr", "rtgs cr",
    "inward", "credited", "payroll",
]
_EMI_KEYWORDS = [
    "emi", "loan emi", "equated monthly", "installment",
    "loan repay", "auto debit", "nach debit", "neft dr",
]
_AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{4,}(?:\.\d{1,2})?)")


def _parse_amount(token: str) -> float:
    return float(token.replace(",", ""))


def extract_bank_statement_data(text: str) -> Dict[str, Any]:
    """Extract monthly income, EMI and DTI from bank statement text."""
    credits: list[float] = []
    debits: list[float] = []

    for line in text.splitlines():
        lower = line.lower()
        amounts = [_parse_amount(m) for m in _AMOUNT_RE.findall(line)]
        amounts = [a for a in amounts if a >= 1000]

        if any(kw in lower for kw in _SALARY_KEYWORDS):
            credits.extend(amounts)
        elif any(kw in lower for kw in _EMI_KEYWORDS):
            debits.extend(amounts)

    income = 0.0
    if credits:
        top = sorted(credits, reverse=True)[:3]
        income = sum(top) / len(top)

    emi = 0.0
    if debits:
        top = sorted(debits, reverse=True)[:3]
        emi = sum(top) / len(top)

    dti = round(emi / income, 4) if income > 0 else None

    return {
        "income": round(income, 2) if income > 0 else None,
        "emi": round(emi, 2) if emi > 0 else None,
        "dti": dti,
    }


# ---------------------------------------------------------------------------
# Credit report parsing
# ---------------------------------------------------------------------------

_SCORE_PATTERNS = [
    re.compile(r"(?:cibil|credit|experian|equifax|transunion)\s*score[:\s]+(\d{3,4})", re.I),
    re.compile(r"current\s+score[:\s]+(\d{3,4})", re.I),
    re.compile(r"score\s+analysis.*?\b(\d{3,4})\b", re.I | re.S),
    re.compile(r"score[:\s]+(\d{3,4})", re.I),
    re.compile(r"(\d{3})\s*/\s*900", re.I),
    re.compile(r"(\d{3})\s*(?:points?|pts)", re.I),
]

_ACTIVE_LOAN_PATTERNS = [
    re.compile(r"active\s+(?:loan|account)s?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"(?:open|live)\s+(?:loan|account)s?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"total\s+active\s+(?:loan|account)s?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"(\d+)\s+active\s+(?:loan|account)s?", re.I),
    re.compile(r"no\.\s*of\s*(?:active\s+)?loans?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"installment\s+(\d+)\s+\$\d[\d,]*", re.I),
    re.compile(r"summary\s+of\s+accounts.*?installment\s+(\d+)", re.I | re.S),
]


def extract_credit_report_data(text: str) -> Dict[str, Any]:
    """Extract credit score and active loan count from credit report text."""
    credit_score: int | None = None
    for pat in _SCORE_PATTERNS:
        match = pat.search(text)
        if match:
            val = int(match.group(1))
            if 300 <= val <= 900:
                credit_score = val
                break

    if credit_score is None:
        analysis_block = re.search(
            r"current\s+score(?P<body>.*?)(?:summary\s+of\s+accounts|alerts)",
            text,
            re.I | re.S,
        )
        if analysis_block:
            paired_scores = re.findall(
                r"\b([3-8]\d{2}|900)\b\s+\b([3-8]\d{2}|900)\b\s+[+-]\d{1,3}\b",
                analysis_block.group("body"),
                re.I,
            )
            if paired_scores:
                credit_score = min(int(current) for current, _projected in paired_scores)

    active_loans: int | None = None
    for pat in _ACTIVE_LOAN_PATTERNS:
        match = pat.search(text)
        if match:
            active_loans = int(match.group(1))
            break

    return {
        "credit_score": credit_score,
        "active_loans": active_loans,
    }


# ---------------------------------------------------------------------------
# Validation / confidence
# ---------------------------------------------------------------------------

_CRITICAL_FIELDS = {
    "bank_statement": ("income", "emi"),
    "credit_report": ("credit_score", "active_loans"),
}

_RECOMMENDED_FIELDS = {
    "bank_statement": ("income", "emi", "dti"),
    "credit_report": ("credit_score", "active_loans"),
}


def validate_document_data(
    doc_type: str,
    data: Dict[str, Any],
    text_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Assess extraction completeness and whether manual review is needed."""
    critical_fields = list(_CRITICAL_FIELDS.get(doc_type, ()))
    recommended_fields = list(_RECOMMENDED_FIELDS.get(doc_type, critical_fields))
    present_fields = [field for field in recommended_fields if data.get(field) is not None]
    missing_critical = [field for field in critical_fields if data.get(field) is None]
    missing_recommended = [field for field in recommended_fields if data.get(field) is None]

    field_ratio = (len(present_fields) / len(recommended_fields)) if recommended_fields else 1.0
    confidence = 0.35 + 0.55 * field_ratio
    if text_metadata.get("char_count", 0) < 250:
        confidence -= 0.20
    if text_metadata.get("ocr_used"):
        confidence -= 0.10
    confidence = max(0.0, min(round(confidence, 2), 1.0))

    warnings = list(text_metadata.get("warnings", []))
    if missing_critical:
        warnings.append(
            "Missing critical extracted fields: " + ", ".join(missing_critical) + "."
        )
    elif missing_recommended:
        warnings.append(
            "Some non-critical fields were not extracted: " + ", ".join(missing_recommended) + "."
        )
    if text_metadata.get("char_count", 0) < 250:
        warnings.append("Very little text was extracted from the PDF; formatting or OCR quality may be limiting accuracy.")

    status = "good"
    if missing_critical:
        status = "insufficient"
    elif missing_recommended:
        status = "partial"

    return {
        "doc_type": doc_type,
        "required_fields": critical_fields,
        "recommended_fields": recommended_fields,
        "present_fields": present_fields,
        "missing_critical_fields": missing_critical,
        "missing_recommended_fields": missing_recommended,
        "confidence": confidence,
        "review_required": bool(missing_critical),
        "status": status,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# LLM-assisted extraction
# ---------------------------------------------------------------------------

def _get_gemini_model():
    import google.generativeai as genai  # noqa: PLC0415

    if not _GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=_GEMINI_API_KEY)
    return genai.GenerativeModel(_GEMINI_MODEL)


def llm_assisted_extract(text: str, doc_type: str) -> Dict[str, Any]:
    """Use Gemini as a recovery path when regex extraction misses critical fields."""
    if not _GEMINI_API_KEY or not text.strip():
        return {}

    fields = {
        "bank_statement": ["income", "emi", "dti"],
        "credit_report": ["credit_score", "active_loans"],
    }.get(doc_type, [])
    if not fields:
        return {}

    prompt = f"""Extract structured financial values from the following {doc_type} text.
Return only valid JSON with these keys: {fields}

Rules:
- Use null when a field is not present.
- Preserve only numeric values.
- For dti, return a decimal ratio such as 0.42, not a percentage string.

Document text:
{text[:12000]}
"""

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
    except Exception:
        return {}

    clean: Dict[str, Any] = {}
    for field in fields:
        value = parsed.get(field)
        if value in ("", "null", None):
            clean[field] = None
            continue
        try:
            clean[field] = int(value) if field in {"credit_score", "active_loans"} else float(value)
        except (TypeError, ValueError):
            clean[field] = None
    return clean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_document(
    file_bytes: bytes,
    doc_type: str | None = None,
) -> Dict[str, Any]:
    """Extract financial data and attach validation metadata."""
    text_metadata = extract_text_bundle(file_bytes)
    text = text_metadata["text"]
    if not text:
        raise ValueError("Could not extract any text from the PDF.")

    resolved_type = doc_type or classify_document(text)

    if resolved_type == "credit_report":
        data = extract_credit_report_data(text)
    else:
        resolved_type = "bank_statement"
        data = extract_bank_statement_data(text)

    validation = validate_document_data(resolved_type, data, text_metadata)

    llm_used = False
    if validation["review_required"]:
        llm_data = llm_assisted_extract(text, resolved_type)
        for key, value in llm_data.items():
            if data.get(key) is None and value is not None:
                data[key] = value
                llm_used = True
        if llm_used:
            validation = validate_document_data(resolved_type, data, text_metadata)
            validation["warnings"].append("LLM-assisted extraction was used to recover missing fields from an unfamiliar document layout.")

    return {
        "doc_type": resolved_type,
        "data": data,
        "validation": validation,
        "text_metadata": {
            **text_metadata,
            "llm_assisted_extraction_used": llm_used,
        },
    }
