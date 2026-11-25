"""Article-aware chunking of a segmented legal text."""

from __future__ import annotations

from dataclasses import dataclass

from easylaw_rag.text.clean import Line
from easylaw_rag.text.lang import AR, Lang

TARGET_CHARS = 1200
MAX_CHARS = 2000
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    kind: str  # "preamble" | "articles" | "body"
    text: str
    articles: list[str]  # article numbers covered ("?" when OCR lost the number)


def article_label(norm_para: str, lang: Lang = AR) -> str | None:
    m = lang.article_re.match(norm_para)
    if not m:
        return None
    num = m.group("num") or "?"
    if num.isdigit():
        return str(int(num))
    return lang.ordinals.get(num, "?")


def _split_long(text: str) -> list[str]:
    """Split an over-long unit on whitespace into overlapping windows."""
    if len(text) <= MAX_CHARS:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + TARGET_CHARS)
        if end < len(text):
            space = text.rfind(" ", start + TARGET_CHARS // 2, end)
            end = space if space > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return [t for t in out if t]


def _pack(units: list[tuple[str, str | None]], kind: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    arts: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append(Chunk(kind, "\n".join(buf), list(arts)))
            buf.clear()
            arts.clear()

    for text, label in units:
        for piece in _split_long(text):
            if buf and sum(map(len, buf)) + len(piece) > TARGET_CHARS:
                flush()
            buf.append(piece)
            if label and label not in arts:
                arts.append(label)
    flush()
    return chunks


def chunk_paragraphs(paras: list[Line], lang: Lang = AR) -> list[Chunk]:
    labels = [article_label(p.norm, lang) for p in paras]
    first = next((i for i, lab in enumerate(labels) if lab), None)
    if first is None:
        return _pack([(p.text, None) for p in paras], "body")

    preamble = [(p.text, None) for p in paras[:first]]
    articles: list[tuple[str, str | None]] = []
    current: list[str] = []
    label: str | None = None
    for p, lab in zip(paras[first:], labels[first:]):
        if lab and current:
            articles.append(("\n".join(current), label))
            current = []
        if lab:
            label = lab
        current.append(p.text)
    if current:
        articles.append(("\n".join(current), label))
    return _pack(preamble, "preamble") + _pack(articles, "articles")
