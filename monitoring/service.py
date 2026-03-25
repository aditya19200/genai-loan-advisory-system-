from __future__ import annotations

from database.sqlite_db import DatabaseManager


class MonitoringService:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def record_prediction(self, request_id: str, latency_ms: float) -> None:
        self.db.insert_monitoring(request_id, prediction_latency_ms=latency_ms, explanation_time_ms=None, explanation_stability=None)

    def record_explanation(self, request_id: str, time_ms: float, stability: float | None) -> None:
        self.db.insert_monitoring(request_id, prediction_latency_ms=None, explanation_time_ms=time_ms, explanation_stability=stability)
