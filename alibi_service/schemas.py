from __future__ import annotations

from pydantic import BaseModel


class CounterfactualRequest(BaseModel):
    request_id: str
    input_payload: dict
    model_artifact_path: str
