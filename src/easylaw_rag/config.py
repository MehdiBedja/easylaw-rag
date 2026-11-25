"""Project configuration. Import this before any Hugging Face library.

- Loads secrets from ``.env`` at the repository root.
- Keeps model downloads inside the repository (``.cache/huggingface``) instead of the user profile.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "512"))
# fp16 is ~3x SLOWER than fp32 on GPUs without tensor cores (e.g. T600 / GTX 16xx); enable it on modern GPUs.
EMBED_FP16 = os.getenv("EMBED_FP16", "0") == "1"


def google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


# One Qdrant collection and one or more processed folders (documents.parquet + chunks.parquet) per language.
COLLECTIONS = {"ar": os.getenv("QDRANT_COLLECTION", "easylaw_ar"), "fr": os.getenv("QDRANT_COLLECTION_FR", "easylaw_fr")}


def corpus_dirs(lang: str) -> list:
    """Processed folders (documents.parquet + chunks.parquet) of a language: the original Arabic scrape plus
    one folder per journal batch built by ``easylaw_rag.pipeline``."""
    family = PROCESSED / ("fr" if lang == "fr" else "ar_journals")
    batches = sorted(d for d in family.glob("*") if (d / "documents.parquet").exists())
    return [PROCESSED, *batches] if lang == "ar" else batches
