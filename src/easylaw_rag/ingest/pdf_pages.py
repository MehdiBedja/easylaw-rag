"""Journal PDF -> reading-order text of every page.

Modern French journals carry a real text layer; older journals (and the Arabic edition, whose text layer
uses broken font encodings) are scans that go through Tesseract (bundled with PyMuPDF, language data in
``.cache/tessdata_*``). In both cases the page is rebuilt from word boxes, because the journal is printed
in two columns and a naive top-to-bottom read interleaves them.

Usage:
    python -m easylaw_rag.ingest.pdf_pages --lang ar --years 1970-1973
    python -m easylaw_rag.ingest.pdf_pages --lang fr --years 2005-2026 --workers 8
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from easylaw_rag.config import DATA, ROOT
from easylaw_rag.ingest.joradp import RAW, parse_years

PAGES = DATA / "interim" / "pages"
OCR_LANG = {"ar": "ara", "fr": "fra"}
# Measured on Arabic 1970 scans, one thread: "best" models 40 s/page, "fast" models 3 s/page, for a small
# gain in accuracy. OCR_QUALITY=best re-reads with the accurate models when time allows (~6 h per 7,000 pages).
OCR_QUALITY = os.getenv("OCR_QUALITY", "fast")
TESSDATA = ROOT / ".cache" / f"tessdata_{OCR_QUALITY}"
OCR_DPI = 300
MIN_TEXT_LAYER_WORDS = 40
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÿ]")
_RULE_RE = re.compile(r"^[|¦\[\]]+$")  # the printed rule between the two columns, read as a "word"

Word = tuple[float, float, float, float, str]  # x0, y0, x1, y1, text


def text_layer_usable(words: list[Word], lang: str) -> bool:
    """True when the embedded text can be trusted (French only: Arabic layers are garbled or absent)."""
    if lang != "fr" or len(words) < MIN_TEXT_LAYER_WORDS:
        return False
    chars = "".join(w[4] for w in words)
    return len(_LATIN_RE.findall(chars)) / max(1, len(chars)) > 0.6


def _rows(words: list[Word]) -> list[list[Word]]:
    """Words grouped into visual rows (which may span both columns)."""
    if not words:
        return []
    tol = 0.5 * statistics.median(w[3] - w[1] for w in words)
    rows: list[list[Word]] = []
    centre = None
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        yc = (w[1] + w[3]) / 2
        if centre is None or yc - centre > tol:
            rows.append([])
            centre = yc
        rows[-1].append(w)
        centre += (yc - centre) / len(rows[-1])
    return [sorted(r, key=lambda w: w[0]) for r in rows]


def _gutter(rows: list[list[Word]], width: float) -> float | None:
    """x position of the gap between the two columns, or None for a single-column page."""
    centres = []
    long_rows = [r for r in rows if len(r) >= 4]
    for r in long_rows:
        gaps = [(b[0] - a[2], (a[2] + b[0]) / 2) for a, b in zip(r, r[1:])]
        gaps = [g for g in gaps if 0.35 * width < g[1] < 0.65 * width and g[0] > 0.02 * width]
        if gaps:
            centres.append(max(gaps)[1])
    if len(long_rows) < 8 or len(centres) < 0.4 * len(long_rows):
        return None
    return statistics.median(centres)


def layout_text(words: list[Word], width: float, rtl: bool) -> str:
    """Reading-order text: full-width rows in place, column rows first column then second."""
    rows = _rows([w for w in words if not _RULE_RE.match(w[4])])
    if not rows:
        return ""
    gutter = _gutter(rows, width)
    pitch = statistics.median([b[0][1] - a[0][1] for a, b in zip(rows, rows[1:])] or [12.0]) or 12.0

    def line(ws: list[Word]) -> tuple[float, str]:
        ordered = sorted(ws, key=lambda w: w[0], reverse=rtl)
        return min(w[1] for w in ws), " ".join(w[4] for w in ordered)

    out: list[str] = []
    cols: tuple[list, list] = ([], [])  # left, right

    def flush() -> None:
        for col in (cols[1], cols[0]) if rtl else cols:
            prev_y = None
            for y, text in col:
                if prev_y is not None and y - prev_y > 1.7 * pitch:
                    out.append("")  # paragraph gap
                out.append(text)
                prev_y = y
            if col:
                out.append("")
            col.clear()

    for r in rows:
        left = [w for w in r if gutter is not None and (w[0] + w[2]) / 2 < gutter]
        right = [w for w in r if gutter is not None and (w[0] + w[2]) / 2 >= gutter]
        straddles = gutter is None or any(w[0] < gutter - 0.01 * width and w[2] > gutter + 0.01 * width for w in r)
        joined = left and right and right[0][0] - left[-1][2] < 0.012 * width
        if straddles or joined:
            flush()
            out.append(line(r)[1])
            continue
        if left:
            cols[0].append(line(left))
        if right:
            cols[1].append(line(right))
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def page_words(page, lang: str) -> tuple[list[Word], str]:
    words = [w[:5] for w in page.get_text("words")]
    if text_layer_usable(words, lang):
        return words, "text"
    tp = page.get_textpage_ocr(language=OCR_LANG[lang], dpi=OCR_DPI, full=True, tessdata=str(TESSDATA))
    return [w[:5] for w in page.get_text("words", textpage=tp)], "ocr"


def journal_out(pdf: Path) -> Path:
    lang, year = pdf.parts[-3], pdf.parts[-2]
    return PAGES / lang / year / f"{pdf.stem}.parquet"


def process_journal(pdf_path: str) -> tuple[str, int, int]:
    """Returns (file name, pages, pages that needed OCR)."""
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)
    pdf = Path(pdf_path)
    lang, year = pdf.parts[-3], int(pdf.parts[-2])
    rows = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc, 1):
            try:
                words, method = page_words(page, lang)
                text = layout_text(words, page.rect.width, rtl=lang == "ar")
            except Exception as e:  # noqa: BLE001 - a damaged page must not lose the whole journal
                text, method = "", f"error: {type(e).__name__}"
            rows.append((lang, year, int(pdf.stem[5:]), i, method, text))
    out = journal_out(pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows, columns=["lang", "year", "number", "page", "method", "text"]).to_parquet(tmp, index=False)
    tmp.replace(out)
    return pdf.name, len(rows), sum(r[4] == "ocr" for r in rows)


def load_pages(lang: str, years: set[int] | None = None) -> pd.DataFrame:
    files = [
        f for f in sorted((PAGES / lang).glob("*/*.parquet")) if years is None or int(f.parent.name) in years
    ]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(OCR_LANG), required=True)
    ap.add_argument("--years", help='e.g. "1970-1973" (default: every downloaded year)')
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    years = parse_years(args.years)
    pdfs = [
        p for p in sorted((RAW / args.lang).glob("*/*.pdf"))
        if (years is None or int(p.parent.name) in years) and not journal_out(p).exists()
    ]  # fmt: skip
    pdfs = pdfs[: args.limit] if args.limit else pdfs
    # Tesseract otherwise starts several OpenMP threads per process and the workers fight for the cores
    # (measured: 39 s/page with 12 workers instead of 7). Must be set before the workers are spawned.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    print(f"{args.lang}: {len(pdfs)} journals to read with {args.workers} workers", flush=True)
    t0, pages, ocr = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_journal, str(p)) for p in pdfs]
        for n, fut in enumerate(as_completed(futures), 1):
            _, n_pages, n_ocr = fut.result()
            pages, ocr = pages + n_pages, ocr + n_ocr
            if n % 10 == 0 or n == len(pdfs):
                rate = pages / (time.perf_counter() - t0)
                eta = (len(pdfs) - n) * (pages / n) / max(rate, 1e-9) / 60
                print(f"  {n}/{len(pdfs)} journals, {pages:,} pages ({ocr:,} OCR), "
                      f"{rate:.1f} pages/s, ~{eta:.0f} min left", flush=True)  # fmt: skip
    print(f"{args.lang}: done, {pages:,} pages in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
