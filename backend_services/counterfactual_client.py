from __future__ import annotations

import time
from typing import Any

import requests

from app.config import ALIBI_SERVICE_URL


class CounterfactualClient:
    def __init__(self, base_url: str = ALIBI_SERVICE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def generate(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        response = requests.post(f"{self.base_url}/counterfactuals/generate", json=payload, timeout=180)
        response.raise_for_status()
        duration_ms = (time.perf_counter() - started) * 1000
        return response.json(), duration_ms
