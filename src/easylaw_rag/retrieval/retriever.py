"""Hybrid retrieval: BGE-M3 dense + sparse search in Qdrant, fused with RRF, then cross-encoder reranking."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from qdrant_client import models

from easylaw_rag import config  # noqa: F401
from easylaw_rag.config import EMBED_FP16, RERANK_MODEL
from easylaw_rag.index.embed import Encoder, embedding_text
from easylaw_rag.index.store import COLLECTION, client

# "84-17" is written "17-84" in much of the OCR (right-to-left digit runs), so search both orders.
# Measured on a 4 GB T600: FlagReranker's default batch (256) pads all pairs together and takes 25 s for
# 40 candidates; batches of 8 give identical scores in 13 s. Fewer candidates is faster but changes the top 8.
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "40"))
# Tokens per (question, passage) pair; the CPU deployment lowers it (and the candidates) to stay responsive.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))

_ARABIC_RE = re.compile(r"[\u0621-\u064a]")


def detect_lang(text: str) -> str:
    """ "ar" when the question is mostly written in Arabic script, else "fr"."""
    letters = [c for c in text if c.isalpha()]
    return "ar" if letters and len(_ARABIC_RE.findall(text)) / len(letters) > 0.4 else "fr"


_LAW_NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4})\s*[-/]\s*(\d{1,4})(?!\d)")


def expand_query(query: str) -> str:
    variants = {f"{b}-{a}" for a, b in _LAW_NUMBER_RE.findall(query)}
    return query if not variants else f"{query} {' '.join(sorted(variants))}"


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict = field(repr=False)

    @property
    def text(self) -> str:
        return self.payload.get("text") or ""


@dataclass
class Filters:
    include_administrative: bool = False
    in_force_only: bool = False
    types: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if not self.include_administrative:
            must.append(models.FieldCondition(key="is_administrative", match=models.MatchValue(value=False)))
        if self.in_force_only:
            must.append(models.FieldCondition(key="in_force", match=models.MatchValue(value=True)))
        if self.types:
            must.append(models.FieldCondition(key="type_text", match=models.MatchAny(any=self.types)))
        if self.year_from or self.year_to:
            must.append(
                models.FieldCondition(key="year", range=models.Range(gte=self.year_from, lte=self.year_to))
            )
        return models.Filter(must=must) if must else None


class Retriever:
    def __init__(self, encoder: Encoder | None = None, use_reranker: bool = True, device: str | None = None):
        """``device``: "cuda", "cpu", or None (= $EASYLAW_DEVICE, else cuda when available)."""
        device = device or os.getenv("EASYLAW_DEVICE") or None
        # Two fp32 XLM-R-large models do not fit in 4 GB of VRAM (they spill into shared memory and stall),
        # so the single-query encoder runs on CPU and the GPU is kept for the 40-pair reranker.
        encoder_device = os.getenv("EASYLAW_ENCODER_DEVICE") or ("cpu" if use_reranker else device)
        self.encoder = encoder or Encoder(device=encoder_device)
        self.qc = client()
        self.last_timings: dict[str, float] = {}
        self.reranker = None
        if use_reranker:
            import torch
            from FlagEmbedding import FlagReranker

            rerank_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.reranker = FlagReranker(
                RERANK_MODEL, use_fp16=EMBED_FP16 and rerank_device != "cpu", devices=rerank_device
            )

    def search(
        self, query: str, k: int = 8, candidates: int = RERANK_CANDIDATES, filters: Filters | None = None,
        collection: str = COLLECTION,
    ) -> list[Hit]:
        t0 = time.perf_counter()
        q = expand_query(query)
        enc = self.encoder.encode([q], batch_size=1)
        t1 = time.perf_counter()
        flt = (filters or Filters()).to_qdrant()
        sparse = enc.sparse[0]
        res = self.qc.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=enc.dense[0], using="dense", limit=candidates, filter=flt),
                models.Prefetch(
                    query=models.SparseVector(indices=list(sparse), values=list(sparse.values())),
                    using="sparse",
                    limit=candidates,
                    filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidates,
            with_payload=True,
        )
        hits = [Hit(str(p.payload["chunk_id"]), p.score, p.payload) for p in res.points]
        t2 = time.perf_counter()
        if self.reranker and hits:
            pairs = [[query, embedding_text(h.payload.get("context", ""), h.text)] for h in hits]
            scores = self.reranker.compute_score(
                pairs, normalize=True, max_length=RERANK_MAX_LENGTH, batch_size=RERANK_BATCH
            )
            for h, s in zip(hits, scores if isinstance(scores, list) else [scores]):
                h.score = float(s)
            hits.sort(key=lambda h: h.score, reverse=True)
        self.last_timings = {"encode": t1 - t0, "search": t2 - t1, "rerank": time.perf_counter() - t2}
        return hits[:k]
