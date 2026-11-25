"""Run the whole journal pipeline for a language and a range of years, each step resumable.

    journal PDFs -> page text (text layer / OCR) -> documents + chunks -> embeddings -> Qdrant

Each batch (language + years) gets its own folders, so batches can be added over time without
re-embedding what is already indexed:
    data/processed/<fr | ar_journals>/<years>/   data/embeddings/<fr | ar_journals>/<years>/

Usage:
    python -m easylaw_rag.pipeline --lang ar --years 1970-1973
    python -m easylaw_rag.pipeline --lang fr --years 2001-2026 --wait-for data/raw/joradp/download.log "fr: finished"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from easylaw_rag.config import COLLECTIONS, DATA, PROCESSED


def batch_dirs(lang: str, years: str) -> tuple[Path, Path]:
    family = "fr" if lang == "fr" else "ar_journals"
    return PROCESSED / family / years, DATA / "embeddings" / family / years


def wait_for(path: Path, text: str, poll: float = 60.0) -> None:
    """Block until ``text`` appears in a log file written by another job (e.g. the download).

    ``"fr: finished#2"`` waits for the second occurrence (the download script logs it once per step).
    """
    text, _, count = text.partition("#")
    times = int(count) if count.isdigit() else 1
    print(f"waiting for {text!r} (x{times}) in {path} ...", flush=True)
    while not (path.exists() and path.read_text(encoding="utf-8", errors="replace").count(text) >= times):
        time.sleep(poll)


def run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-u", "-m", module, *args]
    print(f"\n=== {time.strftime('%H:%M:%S')}  {module} {' '.join(args)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["ar", "fr"], required=True)
    ap.add_argument("--years", required=True, help='e.g. "2001-2026"')
    ap.add_argument("--wait-for", nargs=2, action="append", metavar=("LOG", "TEXT"), default=[],
                    help="start only once TEXT appears in LOG (repeatable)")  # fmt: skip
    ap.add_argument("--ocr-workers", type=int, default=12)
    ap.add_argument("--skip-pages", action="store_true", help="page text is produced by another running job")
    args = ap.parse_args()
    for log, text in args.wait_for:
        wait_for(Path(log), text)

    processed, embeddings = batch_dirs(args.lang, args.years)
    if not args.skip_pages:
        run("easylaw_rag.ingest.pdf_pages", "--lang", args.lang, "--years", args.years, "--workers", str(args.ocr_workers))
    if args.lang == "fr":
        run("easylaw_rag.ingest.joradp_search", "--parse")
    run("easylaw_rag.process.build_journals", "--lang", args.lang, "--years", args.years, "--out", str(processed))
    run("easylaw_rag.index.embed_corpus", "--processed", str(processed), "--out-dir", str(embeddings))
    # --replace-texts: a re-processed text may have fewer chunks than before; drop its old ones first
    run("easylaw_rag.index.store", "--processed", str(processed), "--embed-dir", str(embeddings),
        "--collection", COLLECTIONS[args.lang], "--replace-texts")  # fmt: skip
    print(f"\n=== {time.strftime('%H:%M:%S')}  {args.lang} {args.years}: indexed", flush=True)


if __name__ == "__main__":
    main()
