from __future__ import annotations

from fastapi import FastAPI

from alibi_service.schemas import CounterfactualRequest
from alibi_service.service import AlibiCounterfactualService


service = AlibiCounterfactualService()
app = FastAPI(title="Alibi Counterfactual Service", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/counterfactuals/generate")
def generate_counterfactual(payload: CounterfactualRequest):
    return service.generate(payload.input_payload, payload.model_artifact_path)
