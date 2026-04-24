from __future__ import annotations

from io import BytesIO
from typing import Any

def extract_pdf_text(file_bytes: bytes) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            parts.append(text)
    direct_text = "\n\n".join(parts).strip()
    if len(direct_text) >= 200:
        return {"text": direct_text, "source": "pdf_text", "details": "Native PDF text extraction succeeded."}

    try:
        ocr_text = _extract_pdf_text_with_ocr(file_bytes)
        if len(ocr_text) >= 100:
            return {"text": ocr_text, "source": "ocr_fallback", "details": "OCR fallback extraction succeeded."}
    except Exception as exc:
        return {
            "text": direct_text,
            "source": "unavailable",
            "details": f"Native extraction was weak and OCR fallback failed: {exc}",
        }

    return {
        "text": direct_text,
        "source": "unavailable",
        "details": "PDF text extraction returned insufficient text and OCR fallback did not produce usable content.",
    }


def build_document_rag_query(document_text: str, user_text: str | None = None) -> str:
    snippet = document_text[:4000]
    return (
        f"Uploaded loan document text: {snippet}. "
        f"User context: {user_text or 'Check compliance with RBI borrower-protection policies.'} "
        "Retrieve the most relevant RBI guidelines for document-to-policy compliance review."
    )


def fallback_document_report(filename: str, document_text: str, rag_context: list[dict[str, Any]]) -> dict[str, Any]:
    lowered = document_text.lower()
    satisfied: list[str] = []
    missing: list[str] = []
    keyword_rules = [
        ("Key Fact Statement / KFS", ["key fact statement", "kfs"]),
        ("APR disclosure", ["apr", "annual percentage rate"]),
        ("Cooling-off period", ["cooling-off", "cooling off"]),
        ("Borrower acknowledgment", ["acknowledg", "consent"]),
        ("Grievance redressal contact", ["grievance", "nodal officer", "ombudsman"]),
        ("Direct disbursement wording", ["disbursement", "borrower account"]),
    ]
    for label, patterns in keyword_rules:
        if any(pattern in lowered for pattern in patterns):
            satisfied.append(label)
        else:
            missing.append(label)

    bullets = [
        f"Uploaded document: {filename}.",
        *(f"Satisfied indicator found for: {item}." for item in satisfied[:5]),
        *(f"Potentially missing or unclear in the document: {item}." for item in missing[:5]),
        *(f"Relevant RBI clause retrieved: {item.get('category', 'RBI')} | {item.get('title', 'Guideline')}." for item in rag_context[:3]),
    ]
    summary = (
        "This document-to-policy report checks whether the uploaded loan document appears to contain evidence for key RBI borrower-protection clauses. "
        + ("Potential gaps were found." if missing else "No obvious content gaps were detected in the scanned text.")
    )
    return {
        "title": "Uploaded Document Compliance Report",
        "audience": "User and Loan Officer",
        "summary": summary,
        "bullets": bullets,
    }


def _extract_pdf_text_with_ocr(file_bytes: bytes) -> str:
    import fitz
    import pytesseract
    from PIL import Image

    document = fitz.open(stream=file_bytes, filetype="pdf")
    parts: list[str] = []
    try:
        for page in document:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image)
            text = " ".join(text.split())
            if text:
                parts.append(text)
    finally:
        document.close()
    return "\n\n".join(parts).strip()
