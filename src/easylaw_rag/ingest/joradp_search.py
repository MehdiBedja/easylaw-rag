"""Metadata of every legal text from the joradp.dz search engine (French interface), month by month.

The search engine gives, for each text: its id (shared with the Arabic interface, so both languages align),
type, number, signature date, issuing authority, journal number/date/page and its title.
Raw result pages are saved first (resumable, one request at a time); ``--parse`` turns them into a table.

Usage:
    python -m easylaw_rag.ingest.joradp_search --years 1962-2026     # fetch (polite, resumable)
    python -m easylaw_rag.ingest.joradp_search --parse               # -> data/interim/texts_fr.parquet
"""

from __future__ import annotations

import argparse
import calendar
import html
import re
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote_plus

import httpx
import pandas as pd

from easylaw_rag.config import DATA
from easylaw_rag.ingest.joradp import MAX_CONSECUTIVE_ERRORS, RAW, USER_AGENT, parse_years

SCRIPTS = "http://www.joradp.dz/SCRIPTS"
SEARCH_DIR = RAW / "search" / "fr"
PAGE_SIZE = 200
ENC = "latin-1"

MONTHS_FR = ["janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet", "aout", "septembre", "octobre",
             "novembre", "decembre"]  # fmt: skip

_CLIENT_RE = re.compile(r'name="?Client"?[^>]*value="?(\d+)')
_RANGE_RE = re.compile(r"Texte\s+(\d+)\s+&agrave;\s+(\d+)\s+de\s+(\d+)")
_DVAL_RE = re.compile(r'name="dval"[^>]*value="(\d+)"')
_HEAD_RE = re.compile(
    r'<tr bgcolor="#78a7b9"><td><a href="#(\d+)"[^>]*>.*?JoOpen\("(\d+)","(\d+)","(\d+)","F"\)', re.DOTALL
)
_CELL_RE = re.compile(r"\s*<tr><td colspan=6>(.*?)</td></tr>", re.DOTALL)
_REF_RE = re.compile(r"^(?P<type>.*?)(?:\s+n°\s*(?P<num>\S+))?\s+du\s+(?P<d>\d{1,2})\s+(?P<m>\S+)\s+(?P<y>\d{4})\s*$")
_JO_RE = re.compile(r"JO N°\s*(\d+)\s+du\s+(\d{1,2})\s+(\S+)\s+(\d{4}),\s*Page\s+(\d+)")


class SearchError(RuntimeError):
    pass


class Session:
    """One server-side search session ("Client"): display options are set once, then queries are paged."""

    def __init__(self, delay: float):
        self.delay = delay
        self.http = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=180)
        self.client: str | None = None

    def _post(self, path: str, data: dict) -> str:
        time.sleep(self.delay)
        body = "&".join(f"{k}={quote_plus(v, encoding=ENC)}" for k, v in data.items())
        r = self.http.post(
            f"{SCRIPTS}/Jof_Rec.dll/{path}", content=body.encode(ENC),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )  # fmt: skip
        r.raise_for_status()
        return r.content.decode(ENC)

    def connect(self) -> None:
        r = self.http.get(f"{SCRIPTS}/Jo_Con.dll/Fr")
        r.raise_for_status()
        m = _CLIENT_RE.search(r.content.decode(ENC))
        if not m:
            raise SearchError("no client id in the menu frame")
        self.client = m.group(1)
        # references + summaries, individual decisions included, PAGE_SIZE texts per page, sorted by id
        self._post("OptPost", {"Client": self.client, "dref": "o", "dsom": "o", "ddec": "o",
                               "daff": str(PAGE_SIZE), "dtri": "0 : Sans (Numéro)", "dOri": "r"})  # fmt: skip

    def month(self, year: int, month: int) -> list[str]:
        """All result pages (raw HTML) for the journals published in one month; [] when there are none."""
        if self.client is None:
            self.connect()
        last = calendar.monthrange(year, month)[1]
        page = self._post("RecPost", {
            "Client": self.client, "Start": "0", "zsec": "", "zmin": "", "znat": "", "znjo": "",
            "znjd": f"01/{month:02d}/{year}", "znjf": f"{last:02d}/{month:02d}/{year}",
            "zntx": "", "zntd": "", "zntf": "", "zdes": "", "ztyp": "2",
        })  # fmt: skip
        pages = []
        while True:
            if "Aucun r&eacute;sultat" in page:
                return pages
            rng = _RANGE_RE.search(page)
            if not rng:
                raise SearchError(_message(page))
            pages.append(page)
            if int(rng.group(2)) >= int(rng.group(3)):
                return pages
            dval = _DVAL_RE.search(page)
            page = self._post("AffPost", {"Client": self.client, "dfon": "3", "dval": dval.group(1) if dval else "1"})


def _message(page: str) -> str:
    text = re.sub(r"<style.*?</style>|<[^>]+>", " ", page, flags=re.DOTALL)
    return " ".join(html.unescape(text).split())[:200]


def fetch(years: set[int], delay: float = 2.0, out: Path = SEARCH_DIR) -> None:
    out.mkdir(parents=True, exist_ok=True)
    today = date.today()
    todo = [
        (y, m) for y in sorted(years) for m in range(1, 13)
        if (y, m) <= (today.year, today.month) and not (out / f"{y}-{m:02d}.done").exists()
    ]  # fmt: skip
    print(f"{len(todo)} months to fetch", flush=True)
    session, errors, texts = Session(delay), 0, 0
    for i, (y, m) in enumerate(todo, 1):
        for attempt in range(4):
            try:
                pages = session.month(y, m)
                break
            except (httpx.HTTPError, SearchError) as e:
                wait = 20 * 2**attempt  # includes "Système réinitialisé": the session is rebuilt after the pause
                print(f"  {y}-{m:02d}: {type(e).__name__} {str(e)[:90]} -> retry in {wait}s", flush=True)
                time.sleep(wait)
                session.client = None
        else:
            errors += 1
            if errors >= MAX_CONSECUTIVE_ERRORS:
                print("server keeps failing: stopping. Run the same command later to continue.", flush=True)
                return
            continue
        errors = 0
        for n, page in enumerate(pages, 1):
            (out / f"{y}-{m:02d}-p{n}.html").write_bytes(page.encode(ENC))
        n_texts = sum(len(_HEAD_RE.findall(p)) for p in pages)
        (out / f"{y}-{m:02d}.done").write_text(str(n_texts), "utf-8")
        texts += n_texts
        if i % 12 == 0 or i == len(todo):
            print(f"  {y}-{m:02d}: {i}/{len(todo)} months, {texts:,} texts so far", flush=True)
    print(f"finished: {texts:,} texts", flush=True)


def _clean(cell: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", cell)).split())


def _month(name: str) -> int | None:
    key = name.lower().translate(str.maketrans("éèêûôâî", "eeeuoai")).rstrip(".")
    return MONTHS_FR.index(key) + 1 if key in MONTHS_FR else None


def _date(d: str, m: str, y: str):
    month = _month(m)
    try:
        return pd.Timestamp(int(y), month, int(d)) if month else pd.NaT
    except ValueError:
        return pd.NaT


def parse_page(page: str) -> list[dict]:
    """Top-level texts of a result page (texts nested under 'Modifié par' etc. are listed in their own month)."""
    rows = []
    for head in _HEAD_RE.finditer(page):
        cells, pos = [], head.end()
        pos = page.find("</tr>", pos) + 5
        while m := _CELL_RE.match(page, pos):
            cells.append(_clean(m.group(1)))
            pos = m.end()
        jo_idx = next((i for i, c in enumerate(cells) if _JO_RE.search(c)), None)
        if jo_idx is None or not cells:
            continue
        ref, jo = _REF_RE.match(cells[0]), _JO_RE.search(cells[jo_idx])
        rows.append({
            "id_text": head.group(1),
            "reference": cells[0],
            "type_text": (ref.group("type") if ref else cells[0]).strip(),
            "jt_number": ref.group("num") if ref else None,
            "signature_date": _date(ref.group("d"), ref.group("m"), ref.group("y")) if ref else pd.NaT,
            "source": " ".join(cells[1:jo_idx]) or None,
            "oj_year": int(head.group(2)),
            "oj_number": int(head.group(3)),
            "official_journal_page": int(head.group(4)),
            "publication_date": _date(jo.group(2), jo.group(3), jo.group(4)),
            "description": " ".join(cells[jo_idx + 1 :]) or None,
        })  # fmt: skip
    return rows


def parse(src: Path = SEARCH_DIR, dest: Path = DATA / "interim" / "texts_fr.parquet") -> pd.DataFrame:
    rows = [r for f in sorted(src.glob("*.html")) for r in parse_page(f.read_bytes().decode(ENC))]
    df = pd.DataFrame(rows).drop_duplicates("id_text").sort_values("id_text").reset_index(drop=True)
    df.to_parquet(dest, index=False)
    print(f"{len(df):,} French text records -> {dest}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default=f"1962-{date.today().year}")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--parse", action="store_true", help="only parse the pages already fetched")
    args = ap.parse_args()
    if not args.parse:
        fetch(parse_years(args.years), args.delay)
    parse()


if __name__ == "__main__":
    main()
