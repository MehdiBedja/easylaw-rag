"""Qdrant collection: dense + sparse (BGE-M3) vectors with legal metadata as payload.

Usage:
    python -m easylaw_rag.index.store --recreate     # load all embedded shards into Qdrant
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from easylaw_rag.config import PROCESSED
from easylaw_rag.index.embed_corpus import EMBED_DIR

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "easylaw_ar")
DENSE_DIM = 1024
_NS = uuid.UUID("6f1c2a52-3c1e-4a8e-9a57-2f0d1c6b7e10")

DOC_FIELDS = [
    "id_text", "type_text", "jt_number", "description", "citation", "pdf_url",
    "oj_number", "oj_year", "official_journal_page", "is_administrative", "in_force",
    "amended_by", "repealed_by", "implemented_by", "title_found",
]  # fmt: skip


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=120)


def create_collection(qc: QdrantClient, recreate: bool, collection: str = COLLECTION) -> None:
    if qc.collection_exists(collection):
        if not recreate:
            return
        qc.delete_collection(collection)
    qc.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=True))},
        # int8 quantization keeps the in-RAM index ~4x smaller with negligible recall loss
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
        ),
    )
    for field, schema in [
        ("id_text", models.PayloadSchemaType.KEYWORD),
        ("type_text", models.PayloadSchemaType.KEYWORD),
        ("kind", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER),
        ("in_force", models.PayloadSchemaType.BOOL),
        ("is_administrative", models.PayloadSchemaType.BOOL),
    ]:
        qc.create_payload_index(collection, field, schema)


def _clean(v):
    if isinstance(v, float) and pd.isna(v):
        return None
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def payloads(processed: Path = PROCESSED) -> pd.DataFrame:
    docs = pd.read_parquet(processed / "documents.parquet")
    docs = docs[[c for c in [*DOC_FIELDS, "lang", "publication_date"] if c in docs.columns]]
    docs["year"] = docs["publication_date"].dt.year.astype("Int64")
    docs = docs.drop(columns=["publication_date"])
    chunks = pd.read_parquet(processed / "chunks.parquet")
    # chunks.parquet was built before a citation fix ("قرار رقم nan"); its order is pinned by the
    # embedding manifest, so the header is cleaned here instead of rebuilding the file.
    chunks["context"] = chunks["context"].str.replace(" رقم nan", "", regex=False)
    return chunks.merge(docs, on="id_text", how="left").set_index("chunk_id")


def delete_texts(qc: QdrantClient, collection: str, ids: list[str], batch: int = 500) -> None:
    """Remove every chunk of these texts (before loading a re-processed version of them)."""
    for i in range(0, len(ids), batch):
        cond = models.FieldCondition(key="id_text", match=models.MatchAny(any=ids[i : i + batch]))
        qc.delete(collection, points_selector=models.Filter(must=[cond]), wait=True)


def load(
    embed_dir: Path = EMBED_DIR, batch: int = 256, processed: Path = PROCESSED, collection: str = COLLECTION,
    replace_texts: bool = False,
) -> None:
    qc = client()
    meta = payloads(processed)
    if replace_texts:
        delete_texts(qc, collection, sorted(meta["id_text"].unique()))
    shards = sorted(embed_dir.glob("shard_*.parquet"))
    total, t0 = 0, time.perf_counter()
    for path in shards:
        tab = pq.read_table(path).to_pandas()
        for i in range(0, len(tab), batch):
            part = tab.iloc[i : i + batch]
            points = []
            for r in part.itertuples():
                m = meta.loc[r.chunk_id]
                payload = {"chunk_id": r.chunk_id, **{k: _clean(v) for k, v in m.items()}}
                points.append(
                    models.PointStruct(
                        id=point_id(r.chunk_id),
                        vector={
                            "dense": r.dense.astype("float32").tolist(),
                            "sparse": models.SparseVector(indices=r.sparse_idx.tolist(), values=r.sparse_val.tolist()),
                        },
                        payload=payload,
                    )
                )
            qc.upsert(collection, points=points, wait=False)
            total += len(points)
        print(f"{path.name}: {total:,} points ({total / (time.perf_counter() - t0):.0f}/s)", flush=True)
    print(f"loaded {total:,} points into '{collection}'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--embed-dir", type=Path, default=EMBED_DIR)
    ap.add_argument("--processed", type=Path, default=PROCESSED, help="folder holding chunks.parquet + documents.parquet")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--replace-texts", action="store_true", help="first delete the old chunks of the texts being loaded")
    args = ap.parse_args()
    qc = client()
    create_collection(qc, args.recreate, args.collection)
    load(args.embed_dir, processed=args.processed, collection=args.collection, replace_texts=args.replace_texts)


if __name__ == "__main__":
    main()
