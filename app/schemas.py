from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: str
    job: int = Field(..., ge=0, le=3)
    housing: str
    saving_accounts: str = "unknown"
    checking_account: str = "unknown"
    credit_amount: float = Field(..., gt=0)
    duration: int = Field(..., gt=0)
    purpose: str


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    prediction: int
    probability: float
    explanation_status: str
    created_at: datetime


class ExplanationResponse(BaseModel):
    request_id: str
    status: str
    shap_global: list[dict[str, Any]] = Field(default_factory=list)
    shap_local: list[dict[str, Any]] = Field(default_factory=list)
    lime_local: list[dict[str, Any]] = Field(default_factory=list)
    rule_based_explanation: dict[str, Any] = Field(default_factory=dict)
    eli5_summary: str = ""
    stability_score: float | None = None
    generated_at: datetime | None = None


class FeedbackInput(BaseModel):
    request_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
