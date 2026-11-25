"""Build the retrieval corpus: documents.parquet + chunks.parquet + a processing report.

Usage:
    python -m easylaw_rag.process.build --data data/interim --out data/processed
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from easylaw_rag.process.chunk import chunk_paragraphs
from easylaw_rag.process.metadata import (
    citation,
    is_administrative,
    joradp_pdf_url,
    relation_lists,
)
from easylaw_rag.process.segment import segment, title_key
from easylaw_rag.text.arabic import arabic_words
from easylaw_rag.text.clean import to_lines, to_paragraphs

NEIGHBOR_PRESENT = 0.7  # share of a neighbour's title words that must appear to count as "leaked in"


def _coverage(key: list[str], words: set[str]) -> float:
    return sum(w in words for w in key) / len(key) if key else float("nan")


def process_one(args: tuple[str, str, str | None, list[list[str]]]) -> dict:
    id_text, raw, description, neighbor_keys = args
    lines = to_lines(raw)
    seg = segment(lines, description)
    kept = [l for l in lines[seg.start : seg.end]]
    paras = to_paragraphs(kept)
    chunks = chunk_paragraphs(paras)

    raw_words = set(arabic_words(raw))
    seg_words = {w for p in paras for w in p.norm.split()}
    key = title_key(description)
    return {
        "id_text": id_text,
        "body": "\n".join(p.text for p in paras),
        "title_found": seg.title_found,
        "title_score": seg.title_score,
        "kept_line_share": (seg.end - seg.start) / max(1, len(lines)),
        "own_cov_raw": _coverage(key, raw_words),
        "own_cov_seg": _coverage(key, seg_words),
        "neighbors": len(neighbor_keys),
        "neighbors_raw": sum(_coverage(k, raw_words) >= NEIGHBOR_PRESENT for k in neighbor_keys),
        "neighbors_seg": sum(_coverage(k, seg_words) >= NEIGHBOR_PRESENT for k in neighbor_keys),
        "chunks": [(c.kind, c.text, c.articles) for c in chunks],
    }


def neighbor_keys(jt: pd.DataFrame) -> dict[str, list[list[str]]]:
    """For each text, the *distinctive* title words of the other texts on the same journal page.

    Sibling texts often share most of their title ("يتضمن تفويض الإمضاء إلى ..."), so only words
    absent from our own title can tell whether a neighbour leaked into our segment.
    """
    words = {r.id_text: set(arabic_words(r.description or "")) for r in jt.itertuples()}
    keys = {i: [w for w in title_key(d) if w not in DATE_WORDS] for i, d in zip(jt["id_text"], jt["description"])}
    out: dict[str, list[list[str]]] = {}
    for _, grp in jt.groupby(["official_journal_id", "official_journal_page"]):
        ids = list(grp["id_text"])
        for i in ids:
            out[i] = [
                distinct
                for j in ids
                if j != i and len(distinct := [w for w in keys[j] if w not in words[i]]) >= 3
            ]
    return out


# Dates differ between sibling titles but appear everywhere in bodies: useless for leak detection.
DATE_WORDS = set(
    arabic_words(
        "يناير فبراير مارس أبريل ابريل مايو يونيو يوليو غشت أوت سبتمبر أكتوبر نوفمبر ديسمبر "
        "محرم صفر ربيع الأول الثاني الأولى الثانية جمادى رجب شعبان رمضان شوال القعدة الحجة ذي ذو "
        "عام سنة الموافق المؤرخ مؤرخ أول"
    )
)


def context_header(doc: pd.Series) -> str:
    """Prepended to every chunk before embedding: the clean title survives OCR noise in the body."""
    return f"{doc['citation']}\n{doc['description'] or ''}".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/interim"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--report", type=Path, default=Path("docs/processing_report.md"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None, help="process only N texts (debugging)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    jt = pd.read_parquet(args.data / "juridical_texts.parquet")
    oj = pd.read_parquet(args.data / "official_journals.parquet")
    adj = pd.read_parquet(args.data / "adjustments.parquet")
    if args.limit:
        jt = jt.sample(args.limit, random_state=0)

    # ---------- document metadata ----------
    docs = jt.merge(
        oj.rename(columns={"id": "official_journal_id", "number": "oj_number", "year": "oj_year"})[
            ["official_journal_id", "oj_number", "oj_year"]
        ],
        on="official_journal_id",
        how="left",
    )
    docs["citation"] = [
        citation(r.type_text, r.jt_number, r.signature_date, r.oj_number, r.oj_year, r.official_journal_page)
        for r in docs.itertuples()
    ]
    docs["pdf_url"] = [joradp_pdf_url(y, n) for y, n in zip(docs["oj_year"], docs["oj_number"])]
    docs["is_administrative"] = docs["description"].map(is_administrative)
    docs = docs.merge(relation_lists(adj, docs["id_text"]), on="id_text", how="left")
    docs["in_force"] = docs["repealed_by"].map(len) == 0

    # ---------- segmentation + chunking (parallel) ----------
    has_text = docs["extracted_text"].fillna("").str.strip().str.len() > 0
    nk = neighbor_keys(docs)
    tasks = [
        (r.id_text, r.extracted_text, r.description, nk.get(r.id_text, []))
        for r in docs[has_text].itertuples()
    ]
    print(f"processing {len(tasks):,} texts with {args.workers} workers...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process_one, tasks, chunksize=200))
    res = pd.DataFrame(results)

    docs = docs.merge(res.drop(columns=["chunks"]), on="id_text", how="left")
    docs["has_body"] = docs["body"].fillna("").str.len() > 0

    # ---------- chunks ----------
    meta = docs.set_index("id_text")
    rows = []
    for r in res.itertuples():
        header = context_header(meta.loc[r.id_text])
        for n, (kind, text, articles) in enumerate(r.chunks):
            rows.append((f"{r.id_text}-{n:03d}", r.id_text, n, kind, articles, text, header))
    # texts without OCR still get one metadata-only chunk so they remain findable
    for id_text, d in meta[~meta["has_body"]].iterrows():
        rows.append((f"{id_text}-000", id_text, 0, "metadata", [], "", context_header(d)))
    chunks = pd.DataFrame(rows, columns=["chunk_id", "id_text", "position", "kind", "articles", "text", "context"])

    keep = [
        "id_text", "type_text", "jt_number", "signature_date", "publication_date", "source",
        "description", "oj_number", "oj_year", "official_journal_page", "citation", "pdf_url",
        "is_administrative", "in_force", "amended_by", "amends", "repealed_by", "repeals",
        "implemented_by", "implements", "corrected_by", "corrects", "approved_by", "approves",
        "has_body", "body", "title_found", "title_score",
    ]  # fmt: skip
    docs[keep].to_parquet(args.out / "documents.parquet", index=False)
    chunks.to_parquet(args.out / "chunks.parquet", index=False)

    write_report(docs, chunks, args.report, time.perf_counter() - t0)
    print(f"done in {time.perf_counter() - t0:.0f}s -> {args.out}, report -> {args.report}")


def write_report(docs: pd.DataFrame, chunks: pd.DataFrame, path: Path, seconds: float) -> None:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    t = docs[docs["title_found"].notna()]
    nb = t[t["neighbors"] > 0]
    lens = chunks.loc[chunks["kind"] != "metadata", "text"].str.len()
    kinds = chunks["kind"].value_counts()
    arts = docs[docs["has_body"]].merge(
        chunks.groupby("id_text")["kind"].agg(lambda s: (s == "articles").any()).rename("has_articles"),
        on="id_text",
    )
    lines = [
        "# Processing report\n",
        f"Generated by `python -m easylaw_rag.process.build` in {seconds:.0f}s.\n",
        "## Segmentation\n",
        f"- Texts with OCR: {len(t):,}; title located: **{pct(t['title_found'].mean())}**",
        f"- Share of OCR lines kept after cutting: median {pct(t['kept_line_share'].median())}",
        f"- Own title words present — raw OCR: {pct(t['own_cov_raw'].mean())} → segment: {pct(t['own_cov_seg'].mean())}",
        f"- Texts sharing a page with other texts: {len(nb):,}. Neighbouring texts' titles found inside the "
        f"text — raw OCR: **{nb['neighbors_raw'].sum():,}** → segment: **{nb['neighbors_seg'].sum():,}** "
        f"({pct(1 - nb['neighbors_seg'].sum() / max(1, nb['neighbors_raw'].sum()))} removed)",
        f"  - remaining leaks in texts whose title was located: "
        f"{int(nb.loc[nb['title_found'] == True, 'neighbors_seg'].sum()):,}; "  # noqa: E712
        f"in texts whose title was not found: {int(nb.loc[nb['title_found'] == False, 'neighbors_seg'].sum()):,} "  # noqa: E712
        "(mostly OCR attached to the wrong text by the original scraper — flagged `title_found=False`)\n",
        "## Chunks\n",
        f"- Total chunks: **{len(chunks):,}** "
        + " · ".join(f"{k}: {v:,}" for k, v in kinds.items()),
        f"- Chunk length (chars): median {int(lens.median())}, p95 {int(lens.quantile(0.95))}, max {int(lens.max())}",
        f"- Texts chunked by article: {pct(arts['has_articles'].mean())} of texts with a body\n",
        "## Metadata\n",
        f"- In force (no repeal link): {pct(docs['in_force'].mean())}; with at least one amendment: "
        f"{pct((docs['amended_by'].map(len) > 0).mean())}",
        f"- Administrative (appointments, promotions…): {pct(docs['is_administrative'].mean())}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
