"""Cleanup of OCR'd Journal Officiel pages: drop running headers/noise lines, rebuild paragraphs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from easylaw_rag.text.lang import AR, Lang

_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-zà-ÿ])- (?=[a-zà-ÿ])")  # "disposi- tions" after joining wrapped lines
ARTICLE_RE = AR.article_re  # kept for callers that only deal with Arabic


@dataclass(frozen=True)
class Line:
    text: str  # display text (original script, invisible chars removed)
    norm: str  # normalized text for matching


def is_noise(norm: str, lang: Lang = AR) -> bool:
    """Running headers, page numbers, date banners and OCR crumbs."""
    if not norm:
        return True
    words = norm.split()
    if len(words) <= lang.running_header_max_words and lang.running_header_re.search(norm):
        if lang.code != "ar" or "العدد" in norm or len(words) <= 7:
            return True
    if len(words) <= 10 and lang.date_banner_re.search(norm):
        return True
    # no real word and short: page numbers, table cells, stray glyphs
    if not any(c.isalnum() for c in norm):  # separator rules: "————", "* * *"
        return True
    return not lang.word_re.search(norm) and len(norm) < lang.noise_max_len


def to_lines(raw: str, lang: Lang = AR) -> list[Line | None]:
    """Split raw OCR into lines; blank/noise lines become ``None`` (paragraph separators)."""
    out: list[Line | None] = []
    for chunk in raw.splitlines():
        text = re.sub(r"[ \t]+", " ", _INVISIBLE_RE.sub("", chunk)).strip()
        norm = lang.normalize(text)
        out.append(None if is_noise(norm, lang) else Line(text, norm))
    return out


def to_paragraphs(lines: list[Line | None], lang: Lang = AR) -> list[Line]:
    """Join column-wrapped lines into paragraphs; break on blank lines and structural openings."""
    paras: list[Line] = []
    buf: list[Line] = []

    def flush() -> None:
        if not buf:
            return
        if lang.code == "fr":
            text = _HYPHEN_BREAK_RE.sub("", " ".join(l.text for l in buf))
            paras.append(Line(text, lang.normalize(text)))
        else:
            paras.append(Line(" ".join(l.text for l in buf), " ".join(l.norm for l in buf)))
        buf.clear()

    for line in lines:
        if line is None:
            flush()
            continue
        if buf and (lang.article_re.match(line.norm) or lang.break_before_re.match(line.norm)):
            flush()
        buf.append(line)
    flush()
    return paras
