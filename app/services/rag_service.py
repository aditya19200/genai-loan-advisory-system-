"""RAG service: local embeddings and local/remote Qdrant integration."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_RAG_DIR = BASE_DIR / "artifacts" / "rag"
LOCAL_EMBEDDING_DIR = BASE_DIR / ".hf-cache" / "e5-small"
LOCAL_QDRANT_PATH = ARTIFACTS_RAG_DIR / "qdrant"
LOCAL_GUIDELINES_PATH = ARTIFACTS_RAG_DIR / "rbi_guidelines.json"

EMBEDDING_MODEL = os.environ.get("E5_MODEL", "intfloat/e5-small")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rbi_guidelines")

_embedder = None
_qdrant = None


def _resolve_embedding_source() -> str:
    if LOCAL_EMBEDDING_DIR.exists():
        return str(LOCAL_EMBEDDING_DIR)
    return EMBEDDING_MODEL


def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the RAG dependencies first."
            ) from exc

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        source = _resolve_embedding_source()
        try:
            _embedder = SentenceTransformer(source, local_files_only=True)
        except TypeError:
            _embedder = SentenceTransformer(source)
        except Exception as exc:
            raise RuntimeError(
                "Embedding model is unavailable locally. Cache `intfloat/e5-small` under "
                f"`{LOCAL_EMBEDDING_DIR}` or allow the environment to download it."
            ) from exc
    return _embedder


def get_qdrant_client():
    global _qdrant
    if _qdrant is None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is not installed. Install the RAG dependencies first.") from exc

        try:
            kwargs = {"url": QDRANT_URL}
            if QDRANT_API_KEY:
                kwargs["api_key"] = QDRANT_API_KEY
            _qdrant = QdrantClient(**kwargs)
            _qdrant.get_collections()
        except Exception:
            LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            _qdrant = QdrantClient(path=str(LOCAL_QDRANT_PATH))
    return _qdrant


def create_collection_if_not_exists(vector_size: int) -> None:
    from qdrant_client.http import models as qmodels

    client = get_qdrant_client()
    existing_collections = {collection.name for collection in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing_collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )


def upsert_documents(docs: List[Dict[str, Any]]) -> None:
    """Upsert docs with keys: id, text, metadata."""
    embedder = get_embedder()
    client = get_qdrant_client()

    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    create_collection_if_not_exists(vector_size=embeddings.shape[1])

    from qdrant_client.http import models as qmodels

    points = []
    for doc, emb in zip(docs, embeddings):
        points.append(
            qmodels.PointStruct(
                id=doc.get("id"),
                vector=emb.tolist(),
                payload={"text": doc["text"], **(doc.get("metadata") or {})},
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def bootstrap_local_guidelines() -> int:
    """Seed the local vector store from a JSON file when available."""
    if not LOCAL_GUIDELINES_PATH.exists():
        return 0

    raw = json.loads(LOCAL_GUIDELINES_PATH.read_text(encoding="utf-8"))
    docs = []
    for index, item in enumerate(raw, start=1):
        text = item.get("text")
        if not text:
            continue
        docs.append(
            {
                "id": item.get("id", index),
                "text": text,
                "metadata": {k: v for k, v in item.items() if k not in {"id", "text"}},
            }
        )

    if not docs:
        return 0

    upsert_documents(docs)
    return len(docs)


def query_similar(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    embedder = get_embedder()
    client = get_qdrant_client()

    existing_collections = {collection.name for collection in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing_collections:
        seeded = bootstrap_local_guidelines()
        if seeded == 0:
            raise RuntimeError(
                "RAG knowledge base is not initialized. Add guideline documents to "
                f"`{LOCAL_GUIDELINES_PATH}` or configure a reachable Qdrant collection."
            )

    emb = embedder.encode(query_text, convert_to_numpy=True).tolist()
    response = client.query_points(collection_name=QDRANT_COLLECTION, query=emb, limit=top_k)
    hits = response.points
    results = []
    for hit in hits:
        results.append({"id": hit.id, "score": hit.score, "payload": hit.payload})
    return results
