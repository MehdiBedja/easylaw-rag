"""Chat web app: a single page served by FastAPI, models loaded once in the background.

Usage:
    python -m easylaw_rag.app.server            # then open http://127.0.0.1:8000

Public deployment knobs (environment variables):
    RATE_LIMIT_PER_MIN / RATE_LIMIT_PER_DAY     questions per visitor (default 6 / 60; 0 = unlimited)
    ANSWER_CACHE_SIZE                           identical first questions served from memory (default 256)
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from easylaw_rag.retrieval.retriever import Filters

STATIC = Path(__file__).parent / "static"
RATE_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "6"))
RATE_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "60"))
CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "256"))

_state: dict = {}
_gpu = threading.Lock()  # one question at a time: the reranker needs the whole GPU (or all the CPU cores)
_hits: dict[str, deque] = defaultdict(deque)
_cache: OrderedDict[tuple, dict] = OrderedDict()


def _load() -> None:
    try:
        from easylaw_rag.config import COLLECTIONS
        from easylaw_rag.rag.answer import LegalRAG

        rag = LegalRAG()
        # In the online demo Qdrant restores its snapshots while the models load: wait for both collections.
        deadline = time.time() + float(os.getenv("QDRANT_WAIT_SECONDS", "1800"))
        while True:
            try:
                if all(rag.retriever.qc.collection_exists(c) for c in COLLECTIONS.values()):
                    break
            except Exception:  # Qdrant not listening yet
                pass
            if time.time() > deadline:
                raise RuntimeError("Qdrant collections not available")
            time.sleep(5)
        _state["rag"] = rag
    except Exception as e:  # reported by /api/health instead of silently dying with the thread
        _state["error"] = f"{type(e).__name__}: {e}"
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loading the encoder and the reranker takes 1-3 minutes; serve the page meanwhile.
    threading.Thread(target=_load, daemon=True).start()
    yield
    _state.clear()


app = FastAPI(title="EasyLaw RAG", lifespan=lifespan)


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    history: list[Turn] = Field(default=[], max_length=10)
    in_force_only: bool = False
    include_administrative: bool = False
    lang: str = "auto"  # "auto" follows the language of the question; "ar" / "fr" force a corpus


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ready": "rag" in _state, "error": _state.get("error")}


def _client(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # behind the Hugging Face proxy
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def _check_rate(client: str) -> None:
    now, hits = time.time(), _hits[client]
    while hits and now - hits[0] > 86400:
        hits.popleft()
    if RATE_PER_DAY and len(hits) >= RATE_PER_DAY:
        raise HTTPException(429, "Daily limit reached for this demo / تم بلوغ الحد اليومي. Try again tomorrow.")
    if RATE_PER_MIN and sum(1 for t in hits if now - t < 60) >= RATE_PER_MIN:
        raise HTTPException(429, "Too many questions, wait a minute / أسئلة كثيرة، انتظر دقيقة.")
    hits.append(now)


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> dict:
    if "rag" not in _state:
        raise HTTPException(503, "The service is starting, try again in a minute / الخدمة قيد التشغيل.")
    question = req.question.strip()
    lang = req.lang if req.lang in ("ar", "fr") else None
    key = (question, lang, req.in_force_only, req.include_administrative)
    if not req.history and key in _cache:  # follow-ups depend on the conversation: never cached
        _cache.move_to_end(key)
        return _cache[key]
    _check_rate(_client(request))
    filters = Filters(include_administrative=req.include_administrative, in_force_only=req.in_force_only)
    history = [(t.question, t.answer) for t in req.history]
    try:
        with _gpu:
            ans = _state["rag"].ask(question, filters=filters, history=history, lang=lang)
    except RuntimeError as e:  # every Gemini model failed (overloaded or quota)
        raise HTTPException(503, "The language model is busy, please retry in a moment / النموذج مشغول، أعد المحاولة.") from e
    sources = [
        {**{k: v for k, v in asdict(s).items() if k != "excerpts"}, "cited": s.n in ans.cited, "excerpt": s.excerpts[0][:700] if s.excerpts else ""}
        for s in ans.sources
    ]
    out = {
        "answer": ans.text.strip(),
        "sources": sources,
        "standalone": ans.standalone if ans.standalone != ans.question else "",
        "lang": ans.lang,
        "model": ans.model,
        "seconds": round(ans.seconds, 1),
        "timings": {k: round(v, 1) for k, v in ans.timings.items()},
    }
    if not req.history and CACHE_SIZE:
        _cache[key] = out
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return out


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
