"""Build documents + chunks for a language from whole-journal page text (see ``ingest.pdf_pages``).

Unlike the original scrape (one OCR'd page per text, long texts cut off), each text is read from the page
it starts on up to the page where the next text starts, then cut precisely with its title and the next
title header. Used for the French edition and to fill the years the Arabic scrape left empty.

Usage:
    python -m easylaw_rag.process.build_journals --lang fr --years 2005-2026
    python -m easylaw_rag.process.build_journals --lang ar --years 1970-1973
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from easylaw_rag.config import DATA, PROCESSED, corpus_dirs
from easylaw_rag.ingest.joradp import parse_years
from easylaw_rag.ingest.pdf_pages import load_pages
from easylaw_rag.process.chunk import chunk_paragraphs
from easylaw_rag.process.metadata import (
    RELATION_FIELDS,
    citation_fr,
    is_administrative_fr,
    joradp_pdf_url,
)
from easylaw_rag.process.segment import MIN_BODY_CHARS, locate_title, segment
from easylaw_rag.text.arabic import unreverse_words
from easylaw_rag.text.clean import to_lines, to_paragraphs
from easylaw_rag.text.lang import LANGS

# Until 1993 the site addresses texts by the printed page number, which runs through the whole year;
# from 1994 on by the page of the issue's PDF.
PRINTED_PAGES_UNTIL = 1993
MAX_PAGES = 200  # longest codes fill ~150 journal pages
NEXT_TITLE_MIN_SCORE = 0.6
DOC_COLUMNS = [
    "id_text", "type_text", "jt_number", "signature_date", "publication_date", "source", "description",
    "oj_number", "oj_year", "official_journal_page", "citation", "pdf_url", "is_administrative", "in_force",
    *RELATION_FIELDS, "has_body", "body", "title_found", "title_score", "lang",
]  # fmt: skip


def first_printed_page(page_lines: list[list], docs: list[tuple[str, int, str]], lang) -> int:
    """Printed number of the PDF's first page, chosen so that most titles fall on their expected page."""
    lo = min(p for _, p, _ in docs)
    best, best_score = lo, -1.0
    for base in range(lo, max(0, lo - 4), -1):
        score = 0.0
        for _, page, desc in docs[:15]:
            i = page - base
            if 0 <= i < len(page_lines):
                score += locate_title(page_lines[i], list(dict.fromkeys(lang.words(desc or ""))), lang)[1]
        if score > best_score + 1e-9:
            best, best_score = base, score
    return best


def process_journal(task: tuple[str, int, int, list[str], list[tuple[str, int, str]]]) -> list[dict]:
    lang_code, year, _number, pages, docs = task
    lang = LANGS[lang_code]
    if lang_code == "ar":
        pages = ["\n".join(map(unreverse_words, (t or "").splitlines())) for t in pages]
    page_lines = [to_lines(t or "", lang) for t in pages]
    docs = sorted(docs, key=lambda d: (d[1], d[0]))
    base = first_printed_page(page_lines, docs, lang) if year <= PRINTED_PAGES_UNTIL else 1
    out = []
    for k, (id_text, page, desc) in enumerate(docs):
        s = page - base
        if not 0 <= s < len(pages):
            continue
        later = [p - base for _, p, _ in docs if p > page]
        e = min(min(later) if later else len(pages) - 1, len(pages) - 1, s + MAX_PAGES)
        lines: list = []
        for i in range(s, e + 1):
            lines += [*page_lines[i], None]
        seg = segment(lines, desc, lang, title_limit=len(page_lines[s]))
        # Title not located (OCR damage, unusual layout): keep the whole span from the page the text starts
        # on, still cut at the next text's title when that one can be found. Losing the head of the previous
        # text into the span costs little; keeping only the first page lost the body of long laws.
        start, end = (seg.start, seg.end) if seg.title_found else (0, len(lines))
        if k + 1 < len(docs):  # the next text's own title is a second, independent cut point
            key = list(dict.fromkeys(lang.words(docs[k + 1][2] or "")))
            nxt, score = locate_title(lines[start + 1 :], key, lang)
            if nxt is not None and score >= NEXT_TITLE_MIN_SCORE:
                cut = start + 1 + nxt
                body_chars = sum(len(l.norm) for l in lines[start:cut] if l)
                if lines[cut] and lang.is_header(lines[cut]) and body_chars >= MIN_BODY_CHARS:
                    end = min(end, cut)
        kept = lines[start:end]
        paras = to_paragraphs(kept, lang)
        out.append({
            "id_text": id_text,
            "body": "\n".join(p.text for p in paras),
            "title_found": seg.title_found,
            "title_score": seg.title_score,
            "pages_read": e - s + 1,
            "chunks": [(c.kind, c.text, c.articles) for c in chunk_paragraphs(paras, lang)],
        })  # fmt: skip
    return out


def metadata(lang: str, years: set[int] | None) -> pd.DataFrame:
    """One row per text with the columns of DOC_COLUMNS that do not depend on the page text."""
    ar = pd.read_parquet(PROCESSED / "documents.parquet").drop(columns=["body", "has_body", "title_found", "title_score"])
    if lang == "ar":
        docs = ar
    else:
        fr = pd.read_parquet(DATA / "interim" / "texts_fr.parquet")
        fr["description"] = fr["description"].fillna("")
        docs = fr.merge(ar[["id_text", "is_administrative", "in_force", *RELATION_FIELDS]], on="id_text", how="left")
        for col in RELATION_FIELDS:
            docs[col] = docs[col].map(lambda v: list(v) if v is not None and not isinstance(v, float) else [])
        docs["in_force"] = docs["in_force"].fillna(True).astype(bool)
        docs["is_administrative"] = docs["is_administrative"].fillna(False).astype(bool) | docs["description"].map(
            is_administrative_fr
        )
        docs["citation"] = [
            citation_fr(r.reference, r.oj_number, r.oj_year, r.official_journal_page) for r in docs.itertuples()
        ]
        docs["pdf_url"] = [joradp_pdf_url(y, n, "fr") for y, n in zip(docs["oj_year"], docs["oj_number"])]
    docs["lang"] = lang
    docs["description"] = docs["description"].fillna("")
    docs = docs[docs["oj_year"].notna() & docs["oj_number"].notna() & docs["official_journal_page"].notna()]
    return docs[docs["oj_year"].isin(years)] if years else docs


def context_header(citation_text: str, description: str | None) -> str:
    return f"{citation_text}\n{description or ''}".strip()


def unlocated_ids(lang: str, out: Path) -> set[str]:
    """Texts of the batches already built whose title could not be located (their body is unreliable)."""
    ids: set[str] = set()
    for folder in corpus_dirs(lang):
        if folder.resolve() == out.resolve() or folder == PROCESSED:
            continue
        docs = pd.read_parquet(folder / "documents.parquet", columns=["id_text", "has_body", "title_found"])
        ids.update(docs.loc[docs["has_body"] & (docs["title_found"] == False), "id_text"])  # noqa: E712
    return ids


def build(lang: str, years: set[int] | None, out: Path, workers: int, only: set[str] | None = None) -> None:
    """``only``: rebuild just these texts (their journals are still read whole, to know where each text ends)."""
    t0 = time.perf_counter()
    docs = metadata(lang, years)
    pages = load_pages(lang, years)
    if pages.empty:
        raise SystemExit("no page text yet: run easylaw_rag.ingest.pdf_pages first")
    by_journal = {
        key: grp.sort_values("page")["text"].tolist() for key, grp in pages.groupby(["year", "number"])
    }
    tasks = []
    for (year, number), grp in docs.groupby(["oj_year", "oj_number"]):
        texts = by_journal.get((int(year), int(number)))
        if texts and (only is None or only.intersection(grp["id_text"])):
            listing = list(zip(grp["id_text"], grp["official_journal_page"].astype(int), grp["description"]))
            tasks.append((lang, int(year), int(number), texts, listing))
    covered = sum(len(t[4]) for t in tasks)
    print(f"{lang}: {len(docs):,} texts, {covered:,} in {len(tasks):,} journals with page text", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = pd.DataFrame([r for rows in pool.map(process_journal, tasks, chunksize=8) for r in rows])
    if only is not None:
        res, docs = res[res["id_text"].isin(only)], docs[docs["id_text"].isin(only)]

    docs = docs.merge(res.drop(columns=["chunks"]), on="id_text", how="left")
    docs["has_body"] = docs["body"].fillna("").str.len() > 0
    prefix = "" if lang == "ar" else f"{lang}-"  # Arabic keeps the ids of the original corpus
    meta = docs.set_index("id_text")
    rows = []
    for r in res.itertuples():
        header = context_header(meta.at[r.id_text, "citation"], meta.at[r.id_text, "description"])
        for n, (kind, text, articles) in enumerate(r.chunks):
            rows.append((f"{prefix}{r.id_text}-{n:03d}", r.id_text, n, kind, list(articles), text, header))
    for id_text, d in meta[~meta["has_body"]].iterrows():  # still findable by title
        rows.append((f"{prefix}{id_text}-000", id_text, 0, "metadata", [], "", context_header(d["citation"], d["description"])))
    chunks = pd.DataFrame(rows, columns=["chunk_id", "id_text", "position", "kind", "articles", "text", "context"])

    out.mkdir(parents=True, exist_ok=True)
    docs[DOC_COLUMNS].to_parquet(out / "documents.parquet", index=False)
    chunks.to_parquet(out / "chunks.parquet", index=False)

    with_text = docs[docs["title_found"].notna()]
    lens = chunks.loc[chunks["kind"] != "metadata", "text"].str.len()
    by_article = chunks[chunks["kind"] == "articles"]["id_text"].nunique() / max(1, docs["has_body"].sum())
    print(
        f"{lang}: {len(docs):,} texts, {docs['has_body'].sum():,} with a body; "
        f"title located {100 * with_text['title_found'].mean():.1f}%; "
        f"multi-page {100 * (with_text['pages_read'] > 1).mean():.0f}%; "
        f"{len(chunks):,} chunks (median {int(lens.median())} chars), "
        f"{100 * by_article:.0f}% of texts chunked by article; {time.perf_counter() - t0:.0f}s -> {out}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(LANGS), required=True)
    ap.add_argument("--years", help='e.g. "1970-1973" (default: every year with page text)')
    ap.add_argument("--out", type=Path, help="default: data/processed/<fr | ar_journals>/<years>")
    ap.add_argument("--retry-unlocated", action="store_true",
                    help="rebuild only the texts whose title was not located in the existing batches")  # fmt: skip
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    family = PROCESSED / ("fr" if args.lang == "fr" else "ar_journals")
    out = args.out or family / ("retry-unlocated" if args.retry_unlocated else args.years or "all")
    only = unlocated_ids(args.lang, out) if args.retry_unlocated else None
    build(args.lang, parse_years(args.years), out, args.workers, only)


if __name__ == "__main__":
    main()
