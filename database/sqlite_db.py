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
                    decision TEXT,
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
                    decision TEXT,
                    risk_score REAL,
                    shap_global TEXT,
                    shap_local TEXT,
                    sentiment TEXT,
                    explanation_text TEXT,
                    advisory TEXT,
                    counter_offer TEXT,
                    reports TEXT,
                    explanation_source TEXT,
                    reports_source TEXT,
                    rag_source TEXT,
                    document_rag_source TEXT,
                    document_extraction_source TEXT,
                    document_report_source TEXT,
                    llm_response TEXT,
                    rag_context TEXT,
                    document_rag_context TEXT,
                    uploaded_document_name TEXT,
                    generation_time_ms REAL,
                    generated_at TEXT
                )
                """
            )
            columns = {row["name"] for row in cursor.execute("PRAGMA table_info(explanations)").fetchall()}
            for column_name, column_type in [
                ("decision", "TEXT"),
                ("risk_score", "REAL"),
                ("sentiment", "TEXT"),
                ("explanation_text", "TEXT"),
                ("advisory", "TEXT"),
                ("counter_offer", "TEXT"),
                ("reports", "TEXT"),
                ("explanation_source", "TEXT"),
                ("reports_source", "TEXT"),
                ("rag_source", "TEXT"),
                ("document_rag_source", "TEXT"),
                ("document_extraction_source", "TEXT"),
                ("document_report_source", "TEXT"),
                ("llm_response", "TEXT"),
                ("rag_context", "TEXT"),
                ("document_rag_context", "TEXT"),
                ("uploaded_document_name", "TEXT"),
            ]:
                if column_name not in columns:
                    cursor.execute(f"ALTER TABLE explanations ADD COLUMN {column_name} {column_type}")
            prediction_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(predictions)").fetchall()}
            if "decision" not in prediction_columns:
                cursor.execute("ALTER TABLE predictions ADD COLUMN decision TEXT")
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
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    rag_source TEXT NOT NULL,
                    latency_ms REAL,
                    message_length INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    request_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    extracted_text TEXT NOT NULL,
                    extraction_source TEXT,
                    extraction_details TEXT,
                    uploaded_at TEXT NOT NULL
                )
                """
            )
            uploaded_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(uploaded_documents)").fetchall()}
            for column_name, column_type in [("extraction_source", "TEXT"), ("extraction_details", "TEXT")]:
                if column_name not in uploaded_columns:
                    cursor.execute(f"ALTER TABLE uploaded_documents ADD COLUMN {column_name} {column_type}")
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

    def insert_prediction(self, record: dict[str, Any], explanation_status: str = "pending") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions
                (request_id, model_name, model_version, input_payload, prediction, decision, probability, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["request_id"],
                    record["model_name"],
                    record["model_version"],
                    json.dumps(record["input_payload"]),
                    record["prediction"],
                    record.get("decision"),
                    record["probability"],
                    record["created_at"],
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO explanations (request_id, status)
                VALUES (?, ?)
                """,
                (record["request_id"], explanation_status),
            )

    def insert_explanation(self, request_id: str, payload: dict[str, Any], generation_time_ms: float) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE explanations
                SET status = ?, decision = ?, risk_score = ?, shap_global = ?, shap_local = ?, sentiment = ?,
                    explanation_text = ?, advisory = ?, counter_offer = ?, reports = ?, explanation_source = ?, reports_source = ?,
                    rag_source = ?, document_rag_source = ?, document_extraction_source = ?, document_report_source = ?,
                    llm_response = ?, rag_context = ?, document_rag_context = ?, uploaded_document_name = ?,
                    generation_time_ms = ?, generated_at = ?
                WHERE request_id = ?
                """,
                (
                    "ready",
                    payload["decision"],
                    payload["risk_score"],
                    json.dumps(payload["shap_global"]),
                    json.dumps(payload["shap_local"]),
                    payload["sentiment"],
                    payload["explanation_text"],
                    payload["advisory"],
                    payload["counter_offer"],
                    json.dumps(payload.get("reports", [])),
                    payload.get("explanation_source", "fallback"),
                    payload.get("reports_source", "fallback"),
                    payload.get("rag_source", "unavailable"),
                    payload.get("document_rag_source", "unavailable"),
                    payload.get("document_extraction_source", "unavailable"),
                    payload.get("document_report_source", "fallback"),
                    json.dumps(payload.get("llm_response", {})),
                    json.dumps(payload.get("rag_context", [])),
                    json.dumps(payload.get("document_rag_context", [])),
                    payload.get("uploaded_document_name"),
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
                SET status = ?, explanation_text = ?, generated_at = ?
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

    def insert_chat_metric(self, source: str, rag_source: str, latency_ms: float | None, message_length: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_metrics (source, rag_source, latency_ms, message_length, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source, rag_source, latency_ms, message_length, datetime.now(timezone.utc).isoformat()),
            )

    def upsert_uploaded_document(
        self,
        request_id: str,
        filename: str,
        file_path: str,
        extracted_text: str,
        extraction_source: str,
        extraction_details: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO uploaded_documents (request_id, filename, file_path, extracted_text, extraction_source, extraction_details, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    filename = excluded.filename,
                    file_path = excluded.file_path,
                    extracted_text = excluded.extracted_text,
                    extraction_source = excluded.extraction_source,
                    extraction_details = excluded.extraction_details,
                    uploaded_at = excluded.uploaded_at
                """,
                (
                    request_id,
                    filename,
                    file_path,
                    extracted_text,
                    extraction_source,
                    extraction_details,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def fetch_uploaded_document(self, request_id: str) -> dict[str, Any] | None:
        return self.fetch_one("uploaded_documents", request_id)

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

    def metrics_overview(self) -> dict[str, Any]:
        with self.connect() as conn:
            prediction_rows = [dict(row) for row in conn.execute("SELECT * FROM predictions").fetchall()]
            explanation_rows = [dict(row) for row in conn.execute("SELECT * FROM explanations").fetchall()]
            monitoring_rows = [dict(row) for row in conn.execute("SELECT * FROM monitoring").fetchall()]
            uploaded_rows = [dict(row) for row in conn.execute("SELECT * FROM uploaded_documents").fetchall()]
            feedback_rows = [dict(row) for row in conn.execute("SELECT * FROM feedback").fetchall()]
            chat_rows = [dict(row) for row in conn.execute("SELECT * FROM chat_metrics").fetchall()]

        def average(values: list[float | int | None]) -> float | None:
            usable = [float(value) for value in values if value is not None]
            if not usable:
                return None
            return sum(usable) / len(usable)

        def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for row in rows:
                label = row.get(key) or "unknown"
                counts[label] = counts.get(label, 0) + 1
            return counts

        def json_count(value: str | None) -> int:
            if not value:
                return 0
            try:
                parsed = json.loads(value)
                return len(parsed) if isinstance(parsed, list) else 0
            except Exception:
                return 0

        ready_explanations = [row for row in explanation_rows if row.get("status") == "ready"]
        errored_explanations = [row for row in explanation_rows if row.get("status") == "error"]
        explanation_requested = [row for row in explanation_rows if row.get("status") in {"pending", "ready", "error"}]
        shap_ready = [row for row in ready_explanations if row.get("shap_global") and row.get("shap_local")]
        case_rag_ready = [row for row in ready_explanations if row.get("rag_source") == "retrieved"]
        document_rag_ready = [row for row in ready_explanations if row.get("document_rag_source") == "retrieved"]
        document_reports = [row for row in ready_explanations if row.get("uploaded_document_name")]

        return {
            "volume": {
                "predictions": len(prediction_rows),
                "explanations_requested": len(explanation_requested),
                "explanations_ready": len(ready_explanations),
                "explanations_error": len(errored_explanations),
                "uploaded_documents": len(uploaded_rows),
                "chat_turns": len(chat_rows),
                "feedback_entries": len(feedback_rows),
            },
            "latency": {
                "avg_prediction_latency_ms": average([row.get("prediction_latency_ms") for row in monitoring_rows]),
                "avg_explanation_latency_ms": average([row.get("explanation_time_ms") for row in monitoring_rows]),
                "avg_chat_latency_ms": average([row.get("latency_ms") for row in chat_rows]),
            },
            "explainability": {
                "explanation_success_rate": (len(ready_explanations) / len(explanation_requested)) if explanation_requested else None,
                "shap_coverage_rate": (len(shap_ready) / len(ready_explanations)) if ready_explanations else None,
                "explanation_source_distribution": distribution(ready_explanations, "explanation_source"),
                "reports_source_distribution": distribution(ready_explanations, "reports_source"),
            },
            "rag": {
                "case_rag_success_rate": (len(case_rag_ready) / len(ready_explanations)) if ready_explanations else None,
                "document_rag_success_rate": (len(document_rag_ready) / len(document_reports)) if document_reports else None,
                "avg_case_rag_snippets": average([json_count(row.get("rag_context")) for row in ready_explanations]),
                "avg_document_rag_snippets": average([json_count(row.get("document_rag_context")) for row in document_reports]),
                "document_extraction_distribution": distribution(uploaded_rows, "extraction_source"),
                "document_rag_source_distribution": distribution(document_reports, "document_rag_source"),
            },
            "llm": {
                "explanation_source_distribution": distribution(ready_explanations, "explanation_source"),
                "reports_source_distribution": distribution(ready_explanations, "reports_source"),
                "document_report_source_distribution": distribution(document_reports, "document_report_source"),
                "chat_source_distribution": distribution(chat_rows, "source"),
                "chat_rag_source_distribution": distribution(chat_rows, "rag_source"),
            },
            "feedback": {
                "average_rating": average([row.get("rating") for row in feedback_rows]),
            },
        }
