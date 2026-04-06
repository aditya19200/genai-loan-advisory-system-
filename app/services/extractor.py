"""PDF extraction module for bank statements and credit reports.

Supports:
- Bank Statement: extracts income, EMI, DTI
- Credit Report: extracts credit score, active loan count
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict

import pdfplumber


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return concatenated text from all pages of a PDF."""
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


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
    "credit history", "credit summary",
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
# Matches Indian-style numbers: 1,00,000  or  50000.00  or  1,500
_AMOUNT_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{4,}(?:\.\d{1,2})?)")


def _parse_amount(token: str) -> float:
    return float(token.replace(",", ""))


def extract_bank_statement_data(text: str) -> Dict[str, Any]:
    """
    Extract monthly income, EMI and DTI from bank statement text.

    Strategy:
    - Lines containing salary keywords  → candidate credit amounts
    - Lines containing EMI keywords     → candidate debit amounts
    - Income  = average of top-3 salary credits
    - EMI     = average of top-3 loan debits
    - DTI     = EMI / Income
    """
    credits: list[float] = []
    debits: list[float] = []

    for line in text.splitlines():
        lower = line.lower()
        amounts = [_parse_amount(m) for m in _AMOUNT_RE.findall(line)]
        # Filter out tiny values (likely dates / account numbers)
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

    dti = round(emi / income, 4) if income > 0 else 0.0

    return {
        "income": round(income, 2),
        "emi": round(emi, 2),
        "dti": dti,
    }


# ---------------------------------------------------------------------------
# Credit report parsing
# ---------------------------------------------------------------------------

# Ordered by specificity — first match wins
_SCORE_PATTERNS = [
    re.compile(r"(?:cibil|credit|experian|equifax)\s*score[:\s]+(\d{3,4})", re.I),
    re.compile(r"score[:\s]+(\d{3,4})", re.I),
    re.compile(r"(\d{3})\s*/\s*900", re.I),   # e.g. "720 / 900"
    re.compile(r"(\d{3})\s*(?:points?|pts)", re.I),
]

_ACTIVE_LOAN_PATTERNS = [
    re.compile(r"active\s+(?:loan|account)s?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"(?:open|live)\s+(?:loan|account)s?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"total\s+active\s+(?:loan|account)s?\s*[:\-]\s*(\d+)", re.I),
    re.compile(r"(\d+)\s+active\s+(?:loan|account)s?", re.I),
    re.compile(r"no\.\s*of\s*(?:active\s+)?loans?\s*[:\-]\s*(\d+)", re.I),
]


def extract_credit_report_data(text: str) -> Dict[str, Any]:
    """Extract credit score and active loan count from credit report text."""
    credit_score: int | None = None
    for pat in _SCORE_PATTERNS:
        m = pat.search(text)
        if m:
            val = int(m.group(1))
            if 300 <= val <= 900:
                credit_score = val
                break

    active_loans: int | None = None
    for pat in _ACTIVE_LOAN_PATTERNS:
        m = pat.search(text)
        if m:
            active_loans = int(m.group(1))
            break

    return {
        "credit_score": credit_score,
        "active_loans": active_loans,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_from_document(
    file_bytes: bytes,
    doc_type: str | None = None,
) -> Dict[str, Any]:
    """
    Extract financial data from a PDF document.

    Args:
        file_bytes: Raw PDF bytes.
        doc_type:   'bank_statement' | 'credit_report'.
                    Auto-detected from content when None.

    Returns:
        {
            "doc_type": str,
            "data":     dict   # keys depend on doc_type
        }
    """
    text = extract_text_from_pdf(file_bytes)
    if not text:
        raise ValueError("Could not extract any text from the PDF.")

    resolved_type = doc_type or classify_document(text)

    if resolved_type == "credit_report":
        data = extract_credit_report_data(text)
    else:
        resolved_type = "bank_statement"
        data = extract_bank_statement_data(text)

    return {"doc_type": resolved_type, "data": data}
