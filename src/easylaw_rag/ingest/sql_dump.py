"""Streaming parser for MySQL ``mysqldump`` files.

The EasyLaw dumps are plain ``INSERT INTO `table` VALUES (...),(...);`` statements,
one statement per line. We parse them without needing a MySQL server so the
pipeline stays reproducible on any machine.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_INSERT_RE = re.compile(r"^INSERT INTO `(?P<table>[^`]+)` VALUES ")

# MySQL escape sequences inside single-quoted strings.
_ESCAPES = {
    "0": "\0",
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
    "\\": "\\",
    "'": "'",
    '"': '"',
}

# One token of a VALUES list: a quoted string, NULL, a bare number, or punctuation.
_TOKEN_RE = re.compile(
    r"""
      '(?P<str>(?:[^'\\]|\\.|'')*)'   # quoted string (backslash or doubled-quote escapes)
    | (?P<null>NULL)
    | (?P<num>[-+]?[0-9][0-9.eE+-]*)
    | (?P<open>\()
    | (?P<close>\))
    | (?P<comma>,)
    | (?P<end>;)
    """,
    re.VERBOSE | re.DOTALL,
)

_ESCAPE_RE = re.compile(r"\\(.)|''", re.DOTALL)


def _unescape(raw: str) -> str:
    if "\\" not in raw and "''" not in raw:
        return raw
    return _ESCAPE_RE.sub(
        lambda m: "'" if m.group(0) == "''" else _ESCAPES.get(m.group(1), m.group(1)),
        raw,
    )


def _convert_number(raw: str) -> int | float | str:
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_values(values: str) -> Iterator[tuple]:
    """Yield row tuples from the body of a ``VALUES`` clause."""
    row: list | None = None
    pos = 0
    n = len(values)
    while pos < n:
        m = _TOKEN_RE.match(values, pos)
        if m is None:
            if values[pos].isspace():
                pos += 1
                continue
            snippet = values[pos : pos + 40]
            raise ValueError(f"Unexpected token at offset {pos}: {snippet!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "open":
            row = []
        elif kind == "close":
            if row is None:
                raise ValueError(f"Unbalanced ')' at offset {pos}")
            yield tuple(row)
            row = None
        elif kind == "str":
            row.append(_unescape(m.group("str")))
        elif kind == "null":
            row.append(None)
        elif kind == "num":
            row.append(_convert_number(m.group("num")))
        elif kind == "end":
            return
        # commas need no action


def iter_dump_rows(path: str | Path, table: str | None = None) -> Iterator[tuple]:
    """Stream rows from a mysqldump file, optionally restricted to one table."""
    with open(path, encoding="utf-8", errors="strict", newline="\n") as fh:
        for line in fh:
            m = _INSERT_RE.match(line)
            if not m or (table and m.group("table") != table):
                continue
            yield from parse_values(line[m.end() :].rstrip("\r\n"))
