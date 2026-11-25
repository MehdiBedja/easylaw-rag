"""Polite, resumable download of official-journal PDFs from joradp.dz (Arabic and French editions).

The site is fragile, so this downloads one file at a time with a pause between requests, backs off on
errors and stops when the server looks down. Re-running continues where it stopped.

Usage:
    python -m easylaw_rag.ingest.joradp --lang ar --years 1970-1973
    python -m easylaw_rag.ingest.joradp --lang fr                      # every year in the catalogue
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
import pandas as pd

from easylaw_rag.config import DATA

# https serves a certificate that does not validate; the files are public, so plain http is used.
BASE = "http://www.joradp.dz/FTP"
EDITIONS = {"ar": ("JO-ARABE", "A"), "fr": ("JO-FRANCAIS", "F")}
RAW = DATA / "raw" / "joradp"
USER_AGENT = "easylaw-rag/0.1 (academic project; one request at a time)"
MAX_CONSECUTIVE_ERRORS = 5


def journal_name(lang: str, year: int, number: int) -> str:
    return f"{EDITIONS[lang][1]}{year}{number:03d}.pdf"


def journal_url(lang: str, year: int, number: int) -> str:
    return f"{BASE}/{EDITIONS[lang][0]}/{year}/{journal_name(lang, year, number)}"


def journal_path(lang: str, year: int, number: int, root: Path = RAW) -> Path:
    return root / lang / str(year) / journal_name(lang, year, number)


def parse_years(spec: str | None) -> set[int] | None:
    """ "1970-1973,1991" -> {1970, 1971, 1972, 1973, 1991}"""
    if not spec:
        return None
    years: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.strip().partition("-")
        years.update(range(int(lo), int(hi or lo) + 1))
    return years


def catalogue(years: set[int] | None = None) -> list[tuple[int, int]]:
    """(year, number) of every journal known from the scraped metadata; both editions share the numbering."""
    j = pd.read_parquet(DATA / "interim" / "official_journals.parquet", columns=["year", "number"])
    j = j[j["number"] < 1000].drop_duplicates().sort_values(["year", "number"])
    if years:
        j = j[j["year"].isin(years)]
    return list(zip(j["year"].astype(int), j["number"].astype(int)))


def _missing(root: Path, lang: str) -> tuple[Path, set[str]]:
    """Journals the server answered 404 for: remembered so they are never requested again."""
    path = root / lang / "missing.jsonl"
    names = {json.loads(line)["name"] for line in path.read_text("utf-8").splitlines()} if path.exists() else set()
    return path, names


def download(lang: str, years: set[int] | None, delay: float = 1.5, limit: int | None = None, root: Path = RAW) -> None:
    todo = catalogue(years)
    missing_path, missing = _missing(root, lang)
    todo = [
        (y, n) for y, n in todo
        if journal_name(lang, y, n) not in missing and not journal_path(lang, y, n, root).exists()
    ]  # fmt: skip
    if limit:
        todo = todo[:limit]
    print(f"{lang}: {len(todo)} journals to download ({len(missing)} known missing)", flush=True)
    done = errors = total_bytes = 0
    t0 = time.perf_counter()
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True) as http:
        for year, number in todo:
            name, dest = journal_name(lang, year, number), journal_path(lang, year, number, root)
            for attempt in range(4):
                try:
                    r = http.get(journal_url(lang, year, number))
                except httpx.HTTPError as e:
                    status, body = type(e).__name__, b""
                else:
                    status, body = r.status_code, r.content
                if status == 200 and body[:5] == b"%PDF-":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_suffix(".part")
                    tmp.write_bytes(body)
                    tmp.replace(dest)
                    done, errors, total_bytes = done + 1, 0, total_bytes + len(body)
                    break
                if status == 404:
                    missing_path.parent.mkdir(parents=True, exist_ok=True)
                    with missing_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"name": name, "year": year, "number": number}) + "\n")
                    errors = 0
                    break
                wait = 15 * 2**attempt  # server error or timeout: give the site time to recover
                print(f"  {name}: {status}, retry in {wait}s", flush=True)
                time.sleep(wait)
            else:
                errors += 1
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    print("server keeps failing: stopping. Run the same command later to continue.", flush=True)
                    return
            if (done and done % 25 == 0) or (year, number) == todo[-1]:
                rate = done / (time.perf_counter() - t0)
                print(f"  {name}: {done}/{len(todo)} done, {total_bytes / 1e6:.0f} MB, {rate * 60:.0f}/min", flush=True)
            time.sleep(delay)
    print(f"{lang}: finished, {done} downloaded, {total_bytes / 1e6:.0f} MB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(EDITIONS), required=True)
    ap.add_argument("--years", help='e.g. "1970-1973,1991" (default: all years)')
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--limit", type=int, help="stop after this many journals")
    args = ap.parse_args()
    download(args.lang, parse_years(args.years), args.delay, args.limit)


if __name__ == "__main__":
    main()
