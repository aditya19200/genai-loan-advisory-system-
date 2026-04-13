from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"
RAG_DIR = ARTIFACTS_DIR / "rag"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "database" / "xai_finance.db")))
DATASET_PATH = DATA_DIR / "german_credit.csv"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RBI_DOC_PATH = Path(os.getenv("RBI_DOC_PATH", str(BASE_DIR.parent / "RBI_Guidelines_Updated_2025.docx")))
QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", str(RAG_DIR / "qdrant"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rbi_guidelines")
E5_MODEL = os.getenv("E5_MODEL", "intfloat/e5-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

def normalize_service_url(value: str) -> str:
    return value if value.startswith(("http://", "https://")) else f"http://{value}"


TOP_K_EXPLANATION_FEATURES = 5
BACKGROUND_SAMPLE_SIZE = 50
TRAIN_TEST_RANDOM_STATE = 42
TOP_MODEL_COUNT = 1
LOWER_BORDERLINE_THRESHOLD = 0.40
UPPER_BORDERLINE_THRESHOLD = 0.60


def ensure_directories() -> None:
    for path in [
        DATA_DIR,
        ARTIFACTS_DIR,
        MODELS_DIR,
        METRICS_DIR,
        EXPLANATIONS_DIR,
        RAG_DIR,
        UPLOADS_DIR,
        DOCS_DIR,
        REPORTS_DIR,
        DB_PATH.parent,
    ]:
        path.mkdir(parents=True, exist_ok=True)
