"""Cut a single legal text out of an OCR'd Journal Officiel page.

The OCR of a text usually covers whole pages, so it also contains the tail of the
previous text and the head of the next one. We use the clean ``description``
(the title as published on JORADP) to locate where our text starts, and the next
text header after our signature line (``حرر بالجزائر``) to locate where it ends.
"""

from __future__ import annotations

from dataclasses import dataclass

from easylaw_rag.text.clean import Line
from easylaw_rag.text.lang import AR, Lang

# Arabic patterns under their historical names; other languages come from ``text.lang``.
HEADER_RE = AR.header_re
TITLE_VERB_RE = AR.title_verb_re
AUTHORITY_RE = AR.authority_re

TITLE_KEY_WORDS = 10  # words used by the quality metrics
TITLE_WINDOW = 4  # lines
TITLE_MIN_SCORE = 0.5
SHORT_TITLE_WORDS = 3
MIN_BODY_CHARS = 150


@dataclass(frozen=True)
class Segment:
    start: int  # index into lines (inclusive)
    end: int  # index into lines (exclusive)
    title_found: bool
    title_score: float


def title_key(description: str | None, lang: Lang = AR) -> list[str]:
    return lang.words(description or "")[:TITLE_KEY_WORDS]


HEADER_PREFERENCE_MARGIN = 0.25  # a header-anchored match may score this much lower and still win
TIE_MARGIN = 0.1


def locate_title(
    lines: list[Line | None], key: list[str], lang: Lang = AR, limit: int | None = None
) -> tuple[int | None, float]:
    """Line index where our text starts, found by matching its title against the OCR.

    Bodies often repeat the title ("... في اللجنة الوزارية المشتركة ..."), sometimes more
    legibly than the OCR'd header itself, so a match that sits right after a text header
    is preferred over a slightly better match in running text. ``limit`` restricts the search to
    the first lines (when the page the text starts on is known).
    """
    if not key:
        return None, 0.0
    # One- or two-word titles ("Relative à la santé.") appear all over the body: only a complete match
    # sitting right under a text header counts.
    short = len(key) < SHORT_TITLE_WORDS
    key_set = set(key)
    # tokenised like the key, so that "santé." or "(suite)," still match their word
    words = [set(lang.word_re.findall(l.norm)) if l else set() for l in lines]

    candidates: list[tuple[int, float, bool]] = []  # (start line, score, anchored on a header)
    for i, line in enumerate(lines[:limit]):
        if line is None or not words[i] & key_set:
            continue
        window = set().union(*words[i : i + TITLE_WINDOW])
        score = len(key_set & window) / len(key_set)
        header = next(
            (j for j in range(i, max(-1, i - TITLE_WINDOW), -1) if lines[j] and lang.is_header(lines[j])),
            None,
        )
        candidates.append((i if header is None else header, score, header is not None))

    if short:
        candidates = [c for c in candidates if c[2] and c[1] == 1.0]
    best = max((s for _, s, _ in candidates), default=0.0)
    if best < TITLE_MIN_SCORE:
        return None, best
    floor = max(TITLE_MIN_SCORE, best - HEADER_PREFERENCE_MARGIN)
    anchored = [c for c in candidates if c[2] and c[1] >= floor]
    if anchored:
        top = max(s for _, s, _ in anchored)
        start, score, _ = next(c for c in anchored if c[1] == top)
        return start, score
    start, score, _ = next(c for c in candidates if c[1] >= best - TIE_MARGIN)
    return start, score


def is_title_header(lines: list[Line | None], i: int, lang: Lang = AR) -> bool:
    """A text header ("قرار مؤرخ في ...") immediately followed by its title verb ("يتضمن ...")."""
    line = lines[i]
    if line is None or not lang.is_header(line):
        return False
    following = " ".join(l.norm for l in lines[i : i + TITLE_WINDOW] if l)
    return bool(lang.title_verb_re.search(following))


def locate_end(lines: list[Line | None], start: int, lang: Lang = AR) -> int:
    """Start of the next text: the first title header after a minimal body."""
    chars = 0
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line is None:
            continue
        chars += len(line.norm)
        # NB: we deliberately do NOT cut on a bare "يتضمن/يحدد ..." line below a dated line: visas
        # ("وبمقتضى المرسوم ... الموافق ... الذي / يحدد ...") have exactly that shape. Tried; it truncated
        # 7% of texts mid-preamble. Headers whose type word was destroyed by OCR are left uncut.
        if chars >= MIN_BODY_CHARS and is_title_header(lines, i, lang):
            # include the next text's authority heading ("وزارة ...") in the next text, not ours
            prev = next((j for j in range(i - 1, start, -1) if lines[j] is not None), None)
            if prev is not None and len(lines[prev].norm.split()) <= 8 and lang.authority_re.match(lines[prev].norm):
                return prev
            return i
    return len(lines)


def segment(
    lines: list[Line | None], description: str | None, lang: Lang = AR, title_limit: int | None = None
) -> Segment:
    # match on the full title (deduplicated) so near-identical sibling texts are told apart
    start, score = locate_title(lines, list(dict.fromkeys(lang.words(description or ""))), lang, title_limit)
    if start is None:
        return Segment(0, len(lines), False, score)
    return Segment(start, locate_end(lines, start, lang), True, score)
