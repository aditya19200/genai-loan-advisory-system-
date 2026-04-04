from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from app.config import E5_MODEL, QDRANT_COLLECTION, QDRANT_LOCAL_PATH, RAG_TOP_K, RBI_DOC_PATH


@dataclass
class RagStatus:
    ready: bool
    message: str


class RBIKnowledgeBase:
    def __init__(self) -> None:
        self.doc_path = Path(RBI_DOC_PATH)
        self.qdrant_path = Path(QDRANT_LOCAL_PATH)
        self.manifest_path = self.qdrant_path.parent / "rbi_manifest.json"
        self.collection = QDRANT_COLLECTION
        self._client = None
        self._embedder = None
        self._status = RagStatus(ready=False, message="RAG not initialized")

    def status(self) -> dict[str, Any]:
        return {"ready": self._status.ready, "message": self._status.message, "doc_path": str(self.doc_path)}

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or RAG_TOP_K
        try:
            self._ensure_index()
            if not self._status.ready:
                return []
            embedder = self._get_embedder()
            client = self._get_client()
            vector = embedder.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
            hits = client.search(collection_name=self.collection, query_vector=vector, limit=top_k)
            results: list[dict[str, Any]] = []
            for hit in hits:
                payload = dict(hit.payload or {})
                payload["score"] = float(hit.score)
                results.append(payload)
            return results
        except Exception as exc:
            self._status = RagStatus(ready=False, message=f"RAG retrieval unavailable: {exc}")
            return []

    def _ensure_index(self) -> None:
        if not self.doc_path.exists():
            self._status = RagStatus(ready=False, message=f"RBI document not found at {self.doc_path}")
            return
        signature = self._signature()
        manifest = self._load_manifest()
        client = self._get_client()
        existing_collections = {item.name for item in client.get_collections().collections}
        if manifest.get("signature") == signature and self.collection in existing_collections:
            self._status = RagStatus(ready=True, message="RAG index ready")
            return

        chunks = self._parse_chunks()
        embedder = self._get_embedder()
        vectors = embedder.encode(
            [f"passage: {chunk['category']} | {chunk['title']} | {chunk['text']}" for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        from qdrant_client.http import models as qmodels

        if self.collection in existing_collections:
            client.delete_collection(self.collection)
        client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(size=int(vectors.shape[1]), distance=qmodels.Distance.COSINE),
        )
        points = [
            qmodels.PointStruct(id=index + 1, vector=vector.tolist(), payload=chunk)
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        client.upsert(collection_name=self.collection, points=points)
        self.manifest_path.write_text(json.dumps({"signature": signature, "chunks": len(chunks)}, indent=2), encoding="utf-8")
        self._status = RagStatus(ready=True, message=f"Indexed {len(chunks)} RBI guideline chunks")

    def _parse_chunks(self) -> list[dict[str, Any]]:
        paragraphs = self._extract_docx_paragraphs(self.doc_path)
        chunks: list[dict[str, Any]] = []
        category = "General"
        category_note = ""
        current_rule: dict[str, Any] | None = None
        seen_category = False

        for line in paragraphs:
            if line.startswith("Category "):
                if current_rule:
                    chunks.append(current_rule)
                    current_rule = None
                category = line
                category_note = ""
                seen_category = True
                continue
            if not seen_category:
                continue
            if line.startswith("(") and line.endswith(")"):
                category_note = line
                continue
            if line.startswith("Source:"):
                break

            is_new_rule = ":" in line or line.startswith("•")
            if is_new_rule:
                if current_rule:
                    chunks.append(current_rule)
                title, text = self._split_rule(line)
                current_rule = {
                    "doc_name": self.doc_path.name,
                    "category": category,
                    "category_note": category_note,
                    "title": title,
                    "text": text,
                    "updated": "[UPDATED]" in line,
                }
            elif current_rule:
                current_rule["text"] = f"{current_rule['text']} {line}".strip()

        if current_rule:
            chunks.append(current_rule)
        return chunks

    @staticmethod
    def _extract_docx_paragraphs(path: Path) -> list[str]:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ET.fromstring(xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
            if texts:
                line = "".join(texts).replace("\xa0", " ").strip()
                if line:
                    paragraphs.append(line)
        return paragraphs

    @staticmethod
    def _split_rule(line: str) -> tuple[str, str]:
        cleaned = line.lstrip("•").strip()
        if ":" in cleaned:
            title, remainder = cleaned.split(":", 1)
            return title.strip(), cleaned.strip()
        short_title = re.sub(r"\s+", " ", " ".join(cleaned.split()[:6])).strip()
        return short_title or "Guideline", cleaned

    def _signature(self) -> str:
        stat = self.doc_path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self.qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.qdrant_path))
        return self._client

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(E5_MODEL)
        return self._embedder


_KB: RBIKnowledgeBase | None = None


def get_rbi_knowledge_base() -> RBIKnowledgeBase:
    global _KB
    if _KB is None:
        _KB = RBIKnowledgeBase()
    return _KB


def build_rag_query(
    decision: str,
    risk_score: float,
    shap_local: list[dict[str, Any]],
    user_text: str | None = None,
) -> str:
    top_features = ", ".join(
        item["feature"].replace("num__", "").replace("cat__", "").replace("_", " ")
        for item in shap_local[:4]
    )
    question = user_text or "Explain the decision and relevant RBI borrower protection guidelines."
    return (
        f"Loan decision: {decision}. Risk score: {risk_score:.3f}. "
        f"Main SHAP drivers: {top_features}. "
        f"User question: {question}. "
        "Retrieve RBI borrower protection, KFS, grievance, digital lending, and customer communication guidance relevant to this case."
    )
