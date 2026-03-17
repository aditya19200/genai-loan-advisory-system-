from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import MODEL_REGISTRY_PATH


class ModelRegistry:
    def __init__(self, registry_path: Path | None = None) -> None:
        self.registry_path = registry_path or MODEL_REGISTRY_PATH
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({"models": []}, indent=2), encoding="utf-8")

    def load(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register(self, record: dict[str, Any]) -> None:
        payload = self.load()
        record["registered_at"] = datetime.now(timezone.utc).isoformat()
        payload["models"].append(record)
        payload["models"] = sorted(payload["models"], key=lambda item: item["rank"])
        self.save(payload)

    def top_models(self, limit: int = 3) -> list[dict[str, Any]]:
        payload = self.load()
        return payload["models"][:limit]
