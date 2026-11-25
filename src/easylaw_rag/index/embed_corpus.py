"""Embed every chunk with BGE-M3 (dense + sparse) and store vectors on disk in shards.

Resumable: finished shards are skipped, so the job can be stopped and restarted at any time.
Chunks are processed by priority tier (core legal text first, then preambles, then administrative
texts) so a useful index exists early; within a tier they are sorted by length to minimise padding.
A manifest pins the chunk order: if chunks.parquet is rebuilt, stale shards are detected, never mixed.

Usage:
    python -m easylaw_rag.index.embed_corpus [--batch-size 8] [--shard-size 4096] [--limit N] [--worker i/n]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from easylaw_rag.config import DATA, PROCESSED
from easylaw_rag.index.embed import Encoder, embedding_text

EMBED_DIR = DATA / "embeddings"


TIER_NAMES = {0: "core legal text", 1: "preambles", 2: "administrative texts"}


def load_order(limit: int | None, processed: Path = PROCESSED) -> pd.DataFrame:
    chunks = pd.read_parquet(processed / "chunks.parquet", columns=["chunk_id", "id_text", "kind", "text", "context"])
    admin = pd.read_parquet(processed / "documents.parquet", columns=["id_text", "is_administrative"])
    chunks = chunks.merge(admin, on="id_text", how="left")
    chunks["tier"] = 0
    chunks.loc[chunks["kind"] == "preamble", "tier"] = 1
    chunks.loc[chunks["is_administrative"].fillna(False), "tier"] = 2
    chunks["input"] = [embedding_text(c, t) for c, t in zip(chunks["context"], chunks["text"])]
    chunks["n"] = chunks["input"].str.len()
    chunks = chunks.sort_values(["tier", "n", "chunk_id"], kind="stable").reset_index(drop=True)
    return chunks.head(limit) if limit else chunks


def check_manifest(chunks: pd.DataFrame, shard_size: int, out_dir: Path) -> None:
    digest = hashlib.sha256("\n".join(chunks["chunk_id"]).encode()).hexdigest()
    manifest = {"chunks": len(chunks), "shard_size": shard_size, "order_sha256": digest}
    path = out_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise SystemExit(
                f"{out_dir} holds shards for a different chunk set ({old}). "
                "Move or delete that folder to re-embed from scratch."
            )
    else:
        path.write_text(json.dumps(manifest, indent=2))


def write_shard(path: Path, ids: list[str], dense: list[list[float]], sparse: list[dict[int, float]]) -> None:
    table = pa.table(
        {
            "chunk_id": ids,
            "dense": pa.array([np.asarray(v, dtype=np.float16) for v in dense], type=pa.list_(pa.float16())),
            "sparse_idx": pa.array([list(s.keys()) for s in sparse], type=pa.list_(pa.int32())),
            "sparse_val": pa.array([list(s.values()) for s in sparse], type=pa.list_(pa.float32())),
        }
    )
    tmp = path.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)  # atomic: a shard on disk is always complete


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=EMBED_DIR)
    ap.add_argument("--processed", type=Path, default=PROCESSED, help="folder holding chunks.parquet + documents.parquet")
    ap.add_argument("--worker", default="0/1", help="i/n: process only shards with index %% n == i (multi-GPU)")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_order(args.limit, args.processed)
    check_manifest(chunks, args.shard_size, out_dir)
    n_shards = (len(chunks) + args.shard_size - 1) // args.shard_size
    wi, wn = map(int, args.worker.split("/"))
    todo = [s for s in range(n_shards) if s % wn == wi and not (out_dir / f"shard_{s:05d}.parquet").exists()]
    print(f"{len(chunks):,} chunks in {n_shards} shards; {len(todo)} to do")
    for tier, name in TIER_NAMES.items():
        idx = chunks.index[chunks["tier"] == tier]
        if len(idx):
            print(f"  tier {tier} ({name}): {len(idx):,} chunks, shards "
                  f"{idx.min() // args.shard_size + 1}-{idx.max() // args.shard_size + 1}")
    if not todo:
        return

    encoder = Encoder()
    t_start, done = time.perf_counter(), 0
    for s in todo:
        part = chunks.iloc[s * args.shard_size : (s + 1) * args.shard_size]
        out = encoder.encode(list(part["input"]), batch_size=args.batch_size)
        write_shard(out_dir / f"shard_{s:05d}.parquet", list(part["chunk_id"]), out.dense, out.sparse)
        done += len(part)
        rate = done / (time.perf_counter() - t_start)
        remaining = sum(min(args.shard_size, len(chunks) - t * args.shard_size) for t in todo) - done
        print(f"shard {s + 1}/{n_shards}  {rate:.1f} chunks/s  ETA {remaining / rate / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
