"""Convert the three EasyLaw MySQL dumps into typed Parquet tables.

Usage:
    python -m easylaw_rag.ingest.load_dumps --dumps-dir .. --out data/interim
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from easylaw_rag.ingest.sql_dump import iter_dump_rows

# Column order as it appears in the dumps (verified against the Django migrations).
TABLES = {
    "data_collection_juridicaltext": [
        "id_text",
        "type_text",
        "signature_date",
        "publication_date",
        "jt_number",
        "source",
        "official_journal_page",
        "description",
        "text_file",
        "official_journal_id",
        "extracted_text",
    ],
    "data_collection_officialjournal": ["id", "number", "year", "text_file"],
    "data_collection_adjutstement": ["id", "adjustment_type", "adjusted_num", "adjusting_num"],
}

SHORT_NAMES = {
    "data_collection_juridicaltext": "juridical_texts",
    "data_collection_officialjournal": "official_journals",
    "data_collection_adjutstement": "adjustments",
}


def find_dump(dumps_dir: Path, table: str) -> Path:
    candidates = sorted(dumps_dir.glob(f"lawdata_{table}*.sql"))
    if not candidates:
        raise FileNotFoundError(f"No dump for {table} in {dumps_dir}")
    return candidates[0]


def load_table(path: Path, table: str) -> pd.DataFrame:
    columns = TABLES[table]
    rows = list(iter_dump_rows(path, table))
    bad = [r for r in rows if len(r) != len(columns)]
    if bad:
        raise ValueError(f"{table}: {len(bad)} rows have unexpected arity (e.g. {len(bad[0])})")
    df = pd.DataFrame(rows, columns=columns)
    if table == "data_collection_juridicaltext":
        for col in ("signature_date", "publication_date"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["official_journal_page"] = pd.to_numeric(df["official_journal_page"], errors="coerce").astype("Int32")
        df["official_journal_id"] = pd.to_numeric(df["official_journal_id"], errors="coerce").astype("Int32")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps-dir", type=Path, default=Path(".."))
    ap.add_argument("--out", type=Path, default=Path("data/interim"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for table, short in SHORT_NAMES.items():
        path = find_dump(args.dumps_dir, table)
        t0 = time.perf_counter()
        df = load_table(path, table)
        out = args.out / f"{short}.parquet"
        df.to_parquet(out, index=False)
        print(f"{short:<18} {len(df):>7,} rows  <- {path.name}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
