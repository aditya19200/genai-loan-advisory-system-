"""RAG service: embeddings and Qdrant integration.

This module provides helpers to embed documents (using sentence-transformers) and upsert/search them in Qdrant.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any

EMBEDDING_MODEL = os.environ.get("E5_MODEL", "intfloat/e5-small")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "rbi_guidelines")

_embedder = None
_qdrant = None


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
        try:
            _embedder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
        except TypeError:
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_qdrant_client():
    global _qdrant
    if _qdrant is None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("qdrant-client is not installed. Install the RAG dependencies first.") from exc
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        _qdrant = QdrantClient(**kwargs)
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
    """Docs should be a list of dicts with keys: id (str/int), text (str), metadata (dict)
    This will embed text and upsert to qdrant.
    """
    embedder = get_embedder()
    client = get_qdrant_client()

    texts = [d["text"] for d in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    create_collection_if_not_exists(vector_size=embeddings.shape[1])

    from qdrant_client.http import models as qmodels

    points = []
    for d, emb in zip(docs, embeddings):
        points.append(qmodels.PointStruct(id=d.get("id"), vector=emb.tolist(), payload={"text": d["text"], **(d.get("metadata") or {})}))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def query_similar(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    embedder = get_embedder()
    client = get_qdrant_client()
    emb = embedder.encode(query_text, convert_to_numpy=True).tolist()
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=emb, limit=top_k)
    results = []
    for h in hits:
        results.append({"id": h.id, "score": h.score, "payload": h.payload})
    return results
