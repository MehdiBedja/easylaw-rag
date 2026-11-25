"""Minimal client for any OpenAI-compatible chat endpoint (OpenRouter, Groq, a local gateway ...).

Used for the independent judge of the answer evaluation: a model from another family than the one that
writes the answers. Configured in ``.env``:

    OPENROUTER_API_KEY=...                       # or LLM_API_KEY
    LLM_BASE_URL=https://openrouter.ai/api/v1    # default
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import easylaw_rag.config  # noqa: F401 - loads .env

BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def api_key() -> str:
    key = (os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or LLM_API_KEY) in .env to use an OpenAI-compatible endpoint.")
    return key


def chat(system: str, prompt: str, models: list[str], timeout: float = 90, rate_limit_waits: int = 3) -> tuple[str, str]:
    """Return (text, model used). Models are tried in order; a rate-limited model is retried after a pause."""
    failures = []
    for model in models:
        body = json.dumps({
            "model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }).encode()  # fmt: skip
        for attempt in range(rate_limit_waits + 1):
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
            )  # fmt: skip
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.load(r)
                text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                if text:
                    return text, model
                failures.append(f"{model}: empty ({str(data.get('error'))[:80]})")
                break
            except urllib.error.HTTPError as e:
                detail = e.read()[:160].decode("utf-8", "replace")
                failures.append(f"{model}: {e.code} {detail}")
                if e.code == 429 and "per-day" not in detail and attempt < rate_limit_waits:
                    time.sleep(20)
                    continue
                break
            except (OSError, TimeoutError) as e:
                failures.append(f"{model}: {type(e).__name__}")
                break
    raise RuntimeError("All models failed: " + "; ".join(failures[-4:]))
