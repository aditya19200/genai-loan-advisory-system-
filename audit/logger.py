from __future__ import annotations

import hashlib
import json
from typing import Any

from database.sqlite_db import DatabaseManager


class AuditLogger:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def log(self, request_id: str, model_version: str, payload: dict[str, Any]) -> str:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        self.db.insert_audit_log(request_id, model_version, digest, payload)
        return digest
