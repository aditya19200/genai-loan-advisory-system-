"""Upload router — POST /upload-documents.

Accepts one or more PDF files (bank statement + credit report) together with
the applicant's declared form values, then returns a full RBI compliance report.

Flow:
  1. Validate & read uploaded PDFs
  2. Extract financial data via extractor.py
  3. Compare extracted data with form declarations via comparator.py
  4. Retrieve RBI guidelines from existing RAG pipeline (rag_service)
  5. Generate compliance report via Gemini (compliance_llm.py)
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services import rag_service
from app.services.comparator import compare_with_form
from app.services.compliance_llm import generate_compliance_report
from app.services.extractor import extract_from_document

router = APIRouter(tags=["Document Upload"])

_RAG_QUERY = (
    "RBI loan eligibility rules DTI limits credit score thresholds "
    "income verification guidelines"
)
_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _validate_pdf(upload: UploadFile) -> None:
    """Raise 400 if the upload does not look like a PDF."""
    filename = upload.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is not a PDF. Only .pdf files are accepted.",
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/upload-documents")
async def upload_documents(
    files: List[UploadFile] = File(
        ...,
        description="One or more PDF documents: bank statement and/or credit report.",
    ),
    declared_income: float = Form(
        ...,
        description="Applicant's self-declared monthly income (INR).",
        gt=0,
    ),
    declared_loans: int = Form(
        ...,
        description="Number of active loans declared by the applicant.",
        ge=0,
    ),
) -> Dict[str, Any]:
    """
    Upload financial documents and receive an RBI compliance report.

    **Accepted documents**
    - Bank Statement (PDF) — extracts income, EMI, DTI
    - Credit Report (PDF)  — extracts credit score, active loan count

    **Returns**
    - `extracted_data`       : data parsed from uploaded PDFs
    - `form_data`            : declared values from the request form
    - `comparison`           : mismatch analysis
    - `rag_guidelines_used`  : number of RBI guideline snippets retrieved
    - `compliance_report`    : structured Compliant / Not Compliant verdict
    - `extraction_errors`    : per-file errors (if any)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    for f in files:
        _validate_pdf(f)

    # ------------------------------------------------------------------
    # Step 1 – Extract data from each PDF
    # ------------------------------------------------------------------
    extracted_all: Dict[str, Any] = {}
    detected_doc_types: List[str] = []
    extraction_errors: List[Dict[str, str]] = []

    for upload in files:
        try:
            raw_bytes = await upload.read()
            if len(raw_bytes) == 0:
                raise ValueError("File is empty.")

            result = extract_from_document(raw_bytes)
            detected_doc_types.append(result["doc_type"])
            extracted_all.update(result["data"])  # merge fields from all docs
        except Exception as exc:
            extraction_errors.append(
                {"file": upload.filename or "unknown", "error": str(exc)}
            )

    if not extracted_all and extraction_errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All uploaded files failed text extraction.",
                "errors": extraction_errors,
            },
        )

    extracted_all["_detected_doc_types"] = detected_doc_types

    # ------------------------------------------------------------------
    # Step 2 – Compare extracted vs declared
    # ------------------------------------------------------------------
    form_data: Dict[str, Any] = {
        "declared_income": declared_income,
        "declared_loans": declared_loans,
    }
    comparison = compare_with_form(extracted_all, form_data)

    # ------------------------------------------------------------------
    # Step 3 – Retrieve RBI guidelines via existing RAG pipeline
    # ------------------------------------------------------------------
    rag_contexts: List[Dict[str, Any]] = []
    rag_error: str | None = None
    try:
        rag_contexts = rag_service.query_similar(_RAG_QUERY, top_k=3)
    except Exception as exc:
        rag_error = str(exc)

    # ------------------------------------------------------------------
    # Step 4 – Generate compliance report (Gemini → fallback)
    # ------------------------------------------------------------------
    compliance_report: Dict[str, Any]
    llm_error: str | None = None
    try:
        compliance_report = generate_compliance_report(
            extracted=extracted_all,
            form_data=form_data,
            mismatches=comparison["mismatches"],
            rag_contexts=rag_contexts,
        )
    except Exception as exc:
        llm_error = str(exc)
        compliance_report = {
            "compliance_status": "Error",
            "error": f"Could not generate compliance report: {exc}",
        }

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    response: Dict[str, Any] = {
        "extracted_data": extracted_all,
        "form_data": form_data,
        "comparison": comparison,
        "rag_guidelines_used": len(rag_contexts),
        "compliance_report": compliance_report,
        "extraction_errors": extraction_errors,
    }
    if rag_error:
        response["rag_error"] = rag_error
    if llm_error:
        response["llm_error"] = llm_error

    return response
