"""Benchmark BGE-M3 throughput on this machine with real chunks.

Usage: python scripts/bench_embed.py --n 1000
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from easylaw_rag.config import EMBED_MAX_TOKENS, PROCESSED
from easylaw_rag.index.embed import Encoder, embedding_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch-sizes", default="8,16,32")
    args = ap.parse_args()

    import torch

    chunks = pd.read_parquet(PROCESSED / "chunks.parquet", columns=["kind", "text", "context"])
    sample = chunks[chunks["kind"] != "metadata"].sample(args.n, random_state=0)
    texts = [embedding_text(c, t) for c, t in zip(sample["context"], sample["text"])]

    t0 = time.perf_counter()
    enc = Encoder()
    print(f"model loaded in {time.perf_counter() - t0:.0f}s on {torch.cuda.get_device_name(0)}")

    tok = enc.model.tokenizer
    lens = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
    s = pd.Series(lens)
    print(f"tokens/chunk: median {s.median():.0f}, p95 {s.quantile(.95):.0f}, "
          f"> {EMBED_MAX_TOKENS} (truncated): {(s > EMBED_MAX_TOKENS).mean():.1%}")

    enc.encode(texts[:16])  # warm-up
    for bs in map(int, args.batch_sizes.split(",")):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = enc.encode(texts, batch_size=bs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"batch {bs:>2}: {len(texts) / dt:6.1f} chunks/s, peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB, dim {len(out.dense[0])}, "
              f"sparse terms/chunk {sum(map(len, out.sparse)) / len(out.sparse):.0f}")


if __name__ == "__main__":
    main()
