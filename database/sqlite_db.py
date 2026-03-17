from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.config import DB_PATH


class DatabaseManager:
    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or str(DB_PATH)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    input_payload TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS explanations (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    shap_global TEXT,
                    shap_local TEXT,
                    lime_local TEXT,
                    rule_based_explanation TEXT,
                    eli5_summary TEXT,
                    stability_score REAL,
                    generation_time_ms REAL,
                    generated_at TEXT
                )
                """
            )
            columns = {row["name"] for row in cursor.execute("PRAGMA table_info(explanations)").fetchall()}
            if "rule_based_explanation" not in columns:
                cursor.execute("ALTER TABLE explanations ADD COLUMN rule_based_explanation TEXT")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    audit_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    prediction_latency_ms REAL,
                    explanation_time_ms REAL,
                    explanation_stability REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    comment TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS counterfactuals (
                    request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    result TEXT,
                    error_message TEXT,
                    generation_time_ms REAL,
                    generated_at TEXT
                )
                """
            )

    def insert_prediction(self, record: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions
                (request_id, model_name, model_version, input_payload, prediction, probability, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["request_id"],
                    record["model_name"],
                    record["model_version"],
                    json.dumps(record["input_payload"]),
                    record["prediction"],
                    record["probability"],
                    record["created_at"],
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO explanations (request_id, status)
                VALUES (?, ?)
                """,
                (record["request_id"], "pending"),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO counterfactuals (request_id, status)
                VALUES (?, ?)
                """,
                (record["request_id"], "not_requested"),
            )

    def insert_explanation(self, request_id: str, payload: dict[str, Any], generation_time_ms: float) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE explanations
                SET status = ?, shap_global = ?, shap_local = ?, lime_local = ?, rule_based_explanation = ?, eli5_summary = ?,
                    stability_score = ?, generation_time_ms = ?, generated_at = ?
                WHERE request_id = ?
                """,
                (
                    "ready",
                    json.dumps(payload["shap_global"]),
                    json.dumps(payload["shap_local"]),
                    json.dumps(payload["lime_local"]),
                    json.dumps(payload.get("rule_based_explanation", {})),
                    payload["eli5_summary"],
                    payload["stability_score"],
                    generation_time_ms,
                    datetime.now(timezone.utc).isoformat(),
                    request_id,
                ),
            )

    def mark_explanation_error(self, request_id: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE explanations
                SET status = ?, eli5_summary = ?, generated_at = ?
                WHERE request_id = ?
                """,
                (
                    "error",
                    message,
                    datetime.now(timezone.utc).isoformat(),
                    request_id,
                ),
            )

    def insert_audit_log(self, request_id: str, model_version: str, audit_hash: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (request_id, model_version, audit_hash, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    model_version,
                    audit_hash,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload),
                ),
            )

    def insert_monitoring(self, request_id: str, prediction_latency_ms: float | None, explanation_time_ms: float | None, explanation_stability: float | None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO monitoring (request_id, prediction_latency_ms, explanation_time_ms, explanation_stability, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    prediction_latency_ms,
                    explanation_time_ms,
                    explanation_stability,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def insert_feedback(self, request_id: str, rating: int, comment: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (request_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (request_id, rating, comment, datetime.now(timezone.utc).isoformat()),
            )

    def upsert_counterfactual(self, request_id: str, status: str, result: dict[str, Any] | None = None, error_message: str | None = None, generation_time_ms: float | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO counterfactuals (request_id, status, result, error_message, generation_time_ms, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    status = excluded.status,
                    result = excluded.result,
                    error_message = excluded.error_message,
                    generation_time_ms = excluded.generation_time_ms,
                    generated_at = excluded.generated_at
                """,
                (
                    request_id,
                    status,
                    json.dumps(result) if result is not None else None,
                    error_message,
                    generation_time_ms,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def fetch_one(self, table: str, request_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(f"SELECT * FROM {table} WHERE request_id = ?", (request_id,)).fetchone()
        return dict(row) if row else None

    def fetch_all(self, table: str, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY ROWID DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
