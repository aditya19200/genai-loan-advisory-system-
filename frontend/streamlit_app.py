from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


def normalize_service_url(value: str) -> str:
    return value if value.startswith(("http://", "https://")) else f"http://{value}"


API_BASE = normalize_service_url(os.getenv("API_BASE", "http://127.0.0.1:8000"))

st.set_page_config(page_title="Loan Document Upload", layout="wide")
st.title("Loan Document Upload")
st.caption("Upload bank statements and credit reports, then review extraction quality and compliance output.")


def show_backend_help(error: Exception) -> None:
    st.error("The FastAPI backend is not reachable on `http://127.0.0.1:8000`.")
    st.caption(str(error))
    st.info("Start the API in a separate terminal with `uvicorn app.main:app --reload` from the project root.")


def api_get(path: str) -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE}{path}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        show_backend_help(exc)
        st.stop()


def api_upload(files: list[Any], declared_income: float, declared_loans: int) -> dict[str, Any]:
    payload_files = []
    for uploaded in files:
        payload_files.append(
            (
                "files",
                (uploaded.name, uploaded.getvalue(), uploaded.type or "application/pdf"),
            )
        )

    data = {
        "declared_income": str(declared_income),
        "declared_loans": str(declared_loans),
    }

    try:
        response = requests.post(
            f"{API_BASE}/upload-documents",
            files=payload_files,
            data=data,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        show_backend_help(exc)
        st.stop()


def render_health_banner() -> None:
    health = api_get("/health")
    if health.get("status") == "ok":
        st.success("Backend connected.")
    else:
        st.warning("Backend responded, but health status was unexpected.")


def render_summary_cards(result: dict[str, Any]) -> None:
    extraction_quality = result.get("extraction_quality", {})
    compliance = result.get("compliance_report", {})
    comparison = result.get("comparison", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Extraction Status", extraction_quality.get("status", "unknown").title())
    col2.metric("Extraction Confidence", extraction_quality.get("confidence", "n/a"))
    col3.metric("Mismatch Count", comparison.get("mismatch_count", 0))
    col4.metric("Compliance", compliance.get("compliance_status", "Unknown"))


def render_rag_guidelines(result: dict[str, Any]) -> None:
    guidelines = result.get("rag_guidelines", [])
    if not guidelines:
        return

    st.subheader("Retrieved RBI Guidelines")
    st.caption(f"{len(guidelines)} guideline snippet(s) used for grounding.")

    for item in guidelines:
        payload = item.get("payload") or {}
        title = payload.get("title", "Guideline")
        category = payload.get("category", "RBI")
        score = item.get("score")
        updated = payload.get("updated")
        header = f"{category} | {title}"
        with st.expander(header, expanded=False):
            if payload.get("category_note"):
                st.caption(payload["category_note"])
            if score is not None:
                st.caption(f"Similarity score: {float(score):.3f}")
            if updated:
                st.caption("Includes an updated rule.")
            st.write(payload.get("text", ""))
            if payload.get("source"):
                st.caption(payload["source"])


def render_upload_result(result: dict[str, Any]) -> None:
    render_summary_cards(result)

    if result.get("warnings"):
        for warning in result["warnings"]:
            st.warning(warning)

    if result.get("rag_error"):
        st.info(f"RAG unavailable: {result['rag_error']}")

    compliance = result.get("compliance_report", {})
    status = compliance.get("compliance_status", "Unknown")
    if status == "Compliant":
        st.success("Compliance result: Compliant")
    elif status == "Manual Review Required":
        st.warning("Compliance result: Manual Review Required")
    else:
        st.error(f"Compliance result: {status}")

    render_rag_guidelines(result)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Parsed Data")
        st.json(result.get("extracted_data", {}))

        st.subheader("Compliance Report")
        st.json(compliance)

    with right:
        st.subheader("Extraction Quality")
        st.json(result.get("extraction_quality", {}))

        st.subheader("Mismatch Analysis")
        st.json(result.get("comparison", {}))

    with st.expander("Per-document analysis", expanded=True):
        for item in result.get("document_analysis", []):
            st.markdown(f"**{item.get('file', 'unknown')}**")
            st.json(
                {
                    "doc_type": item.get("doc_type"),
                    "validation": item.get("validation"),
                    "text_metadata": {
                        key: value
                        for key, value in (item.get("text_metadata") or {}).items()
                        if key != "text"
                    },
                    "extracted_fields": item.get("extracted_fields"),
                }
            )

    raw_text_items = []
    for item in result.get("document_analysis", []):
        metadata = item.get("text_metadata") or {}
        text = metadata.get("text")
        if text:
            raw_text_items.append({"file": item.get("file"), "text": text})

    if raw_text_items:
        with st.expander("Raw extracted text", expanded=False):
            for item in raw_text_items:
                st.markdown(f"**{item['file']}**")
                st.text_area(
                    f"Extracted text from {item['file']}",
                    value=item["text"],
                    height=240,
                    disabled=True,
                )


render_health_banner()

tab_upload, tab_api_help = st.tabs(["Document Upload", "API Help"])

with tab_upload:
    st.subheader("Upload and Validate")
    with st.form("upload_form"):
        files = st.file_uploader(
            "Upload one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Use a bank statement and/or a credit report PDF.",
        )
        declared_income = st.number_input(
            "Declared monthly income",
            min_value=1.0,
            value=50000.0,
            step=1000.0,
        )
        declared_loans = st.number_input(
            "Declared active loans",
            min_value=0,
            value=1,
            step=1,
        )
        submitted = st.form_submit_button("Run document analysis", use_container_width=True)

    if submitted:
        if not files:
            st.warning("Upload at least one PDF before submitting.")
        else:
            with st.spinner("Uploading files and generating the compliance report..."):
                result = api_upload(files, declared_income, int(declared_loans))
            st.session_state["latest_upload_result"] = result

    latest_result = st.session_state.get("latest_upload_result")
    if latest_result:
        render_upload_result(latest_result)

with tab_api_help:
    st.subheader("Branch Notes")
    st.write("This UI is intentionally focused on the upload flow supported by `feature/upload`.")
    st.write("It does not try to reuse unsupported pages from `feature/integrated` that depend on different backend APIs.")
    st.code(
        "POST /upload-documents\nGET /health\nGET /docs",
        language="text",
    )
    st.write("Swagger UI remains available at `http://127.0.0.1:8000/docs` for direct API testing.")
