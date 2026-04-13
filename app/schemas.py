from __future__ import annotations

import base64
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    dti: float = Field(..., ge=0, le=2)
    employment_length: float = Field(..., ge=0)
    existing_loans: float = Field(..., ge=0)
    loan_amount: float | None = Field(default=None, gt=0)
    tenure_months: int | None = Field(default=None, gt=0)
    ask_explain: bool = False
    user_text: str | None = None


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    decision: str
    risk_score: float
    explain_available: bool
    explanation_status: str
    created_at: datetime


class ExplanationResponse(BaseModel):
    request_id: str
    status: str
    decision: str | None = None
    risk_score: float | None = None
    shap_global: list[dict[str, Any]] = Field(default_factory=list)
    shap_local: list[dict[str, Any]] = Field(default_factory=list)
    sentiment: str = "neutral"
    explanation_text: str = ""
    advisory: str = ""
    counter_offer: str | None = None
    reports: list[dict[str, Any]] = Field(default_factory=list)
    explanation_source: str = "fallback"
    reports_source: str = "fallback"
    rag_source: str = "unavailable"
    document_rag_source: str = "unavailable"
    document_extraction_source: str = "unavailable"
    document_report_source: str = "fallback"
    llm_response: dict[str, Any] = Field(default_factory=dict)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)
    document_rag_context: list[dict[str, Any]] = Field(default_factory=list)
    uploaded_document_name: str | None = None
    generated_at: datetime | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    applicant_context: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadRequest(BaseModel):
    filename: str = Field(..., min_length=1)
    file_content_base64: str = Field(..., min_length=1)

    def decoded_bytes(self) -> bytes:
        return base64.b64decode(self.file_content_base64)


class ChatResponse(BaseModel):
    reply: str
    source: str
    rag_source: str = "unavailable"
    history: list[ChatMessage] = Field(default_factory=list)
    rag_context: list[dict[str, Any]] = Field(default_factory=list)


class FeedbackInput(BaseModel):
    request_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
