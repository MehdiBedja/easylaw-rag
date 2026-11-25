"""Retrieval evaluation: does the search return the law that answers the question?

Each line of ``eval/questions.jsonl`` is one legal question asked in French and in Arabic, with the texts
that answer it (``"Loi 90-11"``, ``"Ordonnance 75-58"`` ...). The two editions share the site's text ids,
so one list of expected texts serves both languages.

A question is a hit at rank r when the r-th *distinct text* returned is one of the expected texts
(the UI groups chunks by text, so that is what a user sees). Reported per language, for the hybrid
search alone (dense + sparse, RRF) and after reranking:

    hit@1, hit@3, hit@5, hit@10, MRR

Usage:
    python -m easylaw_rag.eval.retrieval                 # prints the table, writes eval/results/
    python -m easylaw_rag.eval.retrieval --langs fr --limit 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from easylaw_rag.config import COLLECTIONS, DATA, ROOT

QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5, 10)
DEPTH = 40  # candidates fetched and reranked, as in the application


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(refs: list[str], texts: pd.DataFrame) -> set[str]:
    """``"Loi 90-11"`` -> ids of the texts with that type and number."""
    ids: set[str] = set()
    for ref in refs:
        kind, number = ref.rsplit(" ", 1)
        found = texts[(texts["jt_number"] == number) & (texts["type_text"] == kind)]["id_text"]
        if found.empty:
            raise ValueError(f"unknown text: {ref}")
        ids.update(found)
    return ids


def first_rank(hits, expected: set[str]) -> int | None:
    """1-based rank, among distinct texts, of the first expected text."""
    seen: list[str] = []
    for h in hits:
        id_text = str(h.payload["id_text"])
        if id_text not in seen:
            seen.append(id_text)
            if id_text in expected:
                return len(seen)
    return None


def summarise(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    out = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in KS}
    out["MRR"] = sum(1 / r for r in ranks if r) / n
    return out


def evaluate(langs: list[str], limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    from easylaw_rag.retrieval.retriever import Retriever

    texts = pd.read_parquet(DATA / "interim" / "texts_fr.parquet", columns=["id_text", "type_text", "jt_number"])
    questions = load_questions()[:limit]
    retriever = Retriever()
    reranker = retriever.reranker
    rows = []
    for n, q in enumerate(questions, 1):
        expected = resolve(q["expected"], texts)
        for lang in langs:
            t0 = time.perf_counter()
            retriever.reranker = None
            hybrid = retriever.search(q[lang], k=DEPTH, candidates=DEPTH, collection=COLLECTIONS[lang])
            retriever.reranker = reranker
            reranked = retriever.search(q[lang], k=DEPTH, candidates=DEPTH, collection=COLLECTIONS[lang])
            rows.append({
                "id": q["id"], "lang": lang, "question": q[lang], "expected": ", ".join(q["expected"]),
                "rank_hybrid": first_rank(hybrid, expected), "rank_reranked": first_rank(reranked, expected),
                "top_text": reranked[0].payload.get("citation") if reranked else None,
                "seconds": round(time.perf_counter() - t0, 1),
            })  # fmt: skip
        print(f"  {n}/{len(questions)} {q['id']}: " + ", ".join(
            f"{r['lang']} {r['rank_hybrid']}->{r['rank_reranked']}" for r in rows[-len(langs):]), flush=True)  # fmt: skip
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"lang": lang, "stage": stage, "questions": len(grp), **summarise([None if pd.isna(r) else int(r) for r in grp[col]])}
        for lang, grp in detail.groupby("lang")
        for stage, col in (("hybrid", "rank_hybrid"), ("hybrid + rerank", "rank_reranked"))
    ])  # fmt: skip
    return detail, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", nargs="+", choices=["ar", "fr"], default=["fr", "ar"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    detail, summary = evaluate(args.langs, args.limit)
    RESULTS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(RESULTS / "retrieval_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "retrieval_summary.csv", index=False)
    print("\n" + summary.round(3).to_string(index=False))
    missed = detail[detail["rank_reranked"].isna() | (detail["rank_reranked"] > 5)]
    if len(missed):
        print("\nnot in the top 5 after reranking:")
        print(missed[["id", "lang", "expected", "rank_hybrid", "rank_reranked", "top_text"]].to_string(index=False))


if __name__ == "__main__":
    main()
