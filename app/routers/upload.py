"""Upload router for POST /upload-documents.

Accepts one or more PDF files together with declared form values and returns
document extraction analysis plus a compliance report.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services import rag_service
from app.services.comparator import compare_with_form
from app.services.compliance_llm import generate_compliance_report
from app.services.extractor import extract_from_document

router = APIRouter(tags=["Document Upload"])


def _build_rag_query(
    extracted_data: Dict[str, Any],
    form_data: Dict[str, Any],
    comparison: Dict[str, Any],
    extraction_quality: Dict[str, Any],
) -> str:
    """Build a case-aware RAG query from the actual upload result."""
    topics: list[str] = [
        "RBI borrower protection",
        "loan disclosure requirements",
        "KFS transparency",
    ]

    detected_types = extracted_data.get("_detected_doc_types") or []
    if "credit_report" in detected_types:
        topics.extend(
            [
                "credit reporting and scores",
                "credit bureau rules",
                "credit bureau borrower rights",
                "credit bureau update rules",
                "free credit report",
                "correction window for credit report errors",
                "borrower alerts on credit inquiries",
                "inquiry alert",
                "reason for rejection",
                "borrower rights on adverse credit decision",
            ]
        )
    if "bank_statement" in detected_types:
        topics.extend(
            [
                "income verification",
                "DTI limits",
                "loan affordability checks",
            ]
        )

    score = extracted_data.get("credit_score")
    if score is not None:
        topics.append(f"credit score {score}")
        topics.append("credit score")
        if score < 650:
            topics.append("low credit score borrower rights and remediation")
            topics.append("adverse credit decision communication")

    active_loans = extracted_data.get("active_loans")
    if active_loans is not None:
        topics.append(f"active loans {active_loans}")

    mismatch_map = comparison.get("mismatches", {})
    if "hidden_loans" in mismatch_map:
        hidden = mismatch_map["hidden_loans"].get("undisclosed_count")
        topics.append(f"undisclosed loans {hidden}")
        topics.append("loan disclosure obligations")
        topics.append("credit report discrepancies")
    if "income_mismatch" in mismatch_map:
        topics.append("income mismatch verification")
    if "high_dti" in mismatch_map:
        topics.append("debt to income limits")
    if "low_credit_score" in mismatch_map:
        topics.append("credit score thresholds and reporting")
        topics.append("credit reporting correction window")
        topics.append("borrower right to know reason for rejection")

    missing_critical = extraction_quality.get("missing_critical_fields") or []
    if missing_critical:
        topics.append("manual review for incomplete borrower documents")

    topics.extend(
        [
            f"declared income {form_data.get('declared_income')}",
            f"declared active loans {form_data.get('declared_loans')}",
        ]
    )

    return ". ".join(dict.fromkeys(topics)) + "."


def _validate_pdf(upload: UploadFile) -> None:
    """Raise 400 if the upload does not look like a PDF."""
    filename = upload.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is not a PDF. Only .pdf files are accepted.",
        )


def _summarize_extraction_quality(document_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse per-document validation into one response-level quality summary."""
    validations = [result["validation"] for result in document_results if result.get("validation")]
    warnings: list[str] = []
    missing_critical_fields: list[str] = []
    low_confidence_docs: list[str] = []

    for result in document_results:
        validation = result.get("validation") or {}
        warnings.extend(validation.get("warnings", []))
        missing_critical_fields.extend(validation.get("missing_critical_fields", []))
        if validation.get("confidence", 1.0) < 0.6:
            low_confidence_docs.append(result.get("file") or "unknown")

    avg_confidence = (
        round(sum(v.get("confidence", 0.0) for v in validations) / len(validations), 2)
        if validations
        else 0.0
    )

    if low_confidence_docs:
        warnings.append(
            "Low-confidence extraction detected for: " + ", ".join(low_confidence_docs) + "."
        )

    return {
        "review_required": any(v.get("review_required") for v in validations),
        "status": (
            "insufficient"
            if any(v.get("status") == "insufficient" for v in validations)
            else ("partial" if any(v.get("status") == "partial" for v in validations) else "good")
        ),
        "confidence": avg_confidence,
        "missing_critical_fields": sorted(set(missing_critical_fields)),
        "warnings": sorted(set(warnings)),
        "documents": [
            {
                "file": result.get("file"),
                "doc_type": result.get("doc_type"),
                "status": result.get("validation", {}).get("status"),
                "confidence": result.get("validation", {}).get("confidence"),
                "review_required": result.get("validation", {}).get("review_required"),
            }
            for result in document_results
        ],
    }


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
    Upload financial documents and receive extraction analysis plus a compliance report.

    Returns:
    - merged extracted data
    - per-document extraction analysis
    - overall extraction-quality summary
    - mismatch analysis against declared form values
    - compliance report with manual-review support
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    for upload in files:
        _validate_pdf(upload)

    extracted_all: Dict[str, Any] = {}
    document_analysis: List[Dict[str, Any]] = []
    detected_doc_types: List[str] = []
    extraction_errors: List[Dict[str, str]] = []

    for upload in files:
        try:
            raw_bytes = await upload.read()
            if len(raw_bytes) == 0:
                raise ValueError("File is empty.")

            result = extract_from_document(raw_bytes)
            detected_doc_types.append(result["doc_type"])
            extracted_all.update(result["data"])
            document_analysis.append(
                {
                    "file": upload.filename or "unknown",
                    "doc_type": result["doc_type"],
                    "validation": result["validation"],
                    "text_metadata": result["text_metadata"],
                    "extracted_fields": result["data"],
                }
            )
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
    extraction_quality = _summarize_extraction_quality(document_analysis)

    form_data: Dict[str, Any] = {
        "declared_income": declared_income,
        "declared_loans": declared_loans,
    }
    comparison = compare_with_form(extracted_all, form_data)

    rag_contexts: List[Dict[str, Any]] = []
    rag_error: str | None = None
    try:
        rag_query = _build_rag_query(extracted_all, form_data, comparison, extraction_quality)
        rag_contexts = rag_service.query_similar(rag_query, top_k=3)
    except Exception as exc:
        rag_error = str(exc)
        rag_query = ""

    compliance_report: Dict[str, Any]
    llm_error: str | None = None
    try:
        compliance_report = generate_compliance_report(
            extracted=extracted_all,
            form_data=form_data,
            mismatches=comparison["mismatches"],
            rag_contexts=rag_contexts,
            extraction_assessment=extraction_quality,
        )
    except Exception as exc:
        llm_error = str(exc)
        compliance_report = {
            "compliance_status": "Error",
            "error": f"Could not generate compliance report: {exc}",
            "manual_review_required": extraction_quality["review_required"],
            "warnings": extraction_quality["warnings"],
        }

    response: Dict[str, Any] = {
        "extracted_data": extracted_all,
        "document_analysis": document_analysis,
        "extraction_quality": extraction_quality,
        "form_data": form_data,
        "comparison": comparison,
        "rag_query": rag_query,
        "rag_guidelines_used": len(rag_contexts),
        "rag_guidelines": rag_contexts,
        "compliance_report": compliance_report,
        "extraction_errors": extraction_errors,
        "review_required": extraction_quality["review_required"],
        "warnings": extraction_quality["warnings"],
    }
    if rag_error:
        response["rag_error"] = rag_error
    if llm_error:
        response["llm_error"] = llm_error

    return response
