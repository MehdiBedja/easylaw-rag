"""Arabic text utilities: normalization for matching/search and quality signals."""

from __future__ import annotations

import re
import unicodedata

# Harakat, tanween, shadda, sukun, superscript alef, Quranic marks.
_DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
# Zero-width / bidi control characters frequently left behind by OCR and copy-paste.
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")

_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        # Arabic-Indic and Persian digits -> ASCII
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
    }
)

ARABIC_LETTER_RE = re.compile(r"[ء-غف-ي]")
ARABIC_WORD_RE = re.compile(r"[ء-غف-ي]{2,}")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
MOJIBAKE_RE = re.compile(r"Ø[\u0080-ÿ§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿]|Ù[\u0080-ÿ]")


def normalize(text: str) -> str:
    """Aggressive normalization used for matching and lexical search (not for display)."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _DIACRITICS_RE.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_CHAR_MAP)
    return re.sub(r"\s+", " ", text).strip()


def arabic_words(text: str) -> list[str]:
    """Normalized Arabic word tokens (length >= 2)."""
    return ARABIC_WORD_RE.findall(normalize(text))


# Tesseract sometimes reads a run of Arabic words left-to-right, giving each word with its letters
# reversed ("ىضتقمبو رمالا" for "وبمقتضى الامر"). Two letters only ever end a word (ى، ة), so a word that
# starts with one is reversed for sure; a long word ending in "لا" is very likely a reversed "ال...".
_WORD_FINAL_ONLY = ("ى", "ة")


def _looks_reversed(tok: str) -> int:
    """2 = certainly reversed, 1 = probably, 0 = no sign."""
    if len(tok) >= 2 and tok[0] in _WORD_FINAL_ONLY and ARABIC_LETTER_RE.search(tok[1:]):
        return 2
    return int(len(tok) >= 5 and tok.endswith("لا") and not tok.startswith("ال"))


def unreverse_words(line: str) -> str:
    """Put reversed OCR words back in reading order; lines without clear evidence are returned unchanged."""
    toks = line.split(" ")
    marks = [_looks_reversed(t) for t in toks]
    if 2 not in marks and sum(marks) < 2:
        return line
    flagged = [i for i, m in enumerate(marks) if m]
    out = list(toks)
    for i in range(flagged[0], flagged[-1] + 1):
        # words caught between two reversed words were read in the same pass ("ىف" -> "فى", "نم" -> "من")
        near = marks[i] or any(marks[j] for j in (i - 1, i + 1) if 0 <= j < len(toks))
        if near and ARABIC_LETTER_RE.search(toks[i]) and not toks[i].startswith("ال"):
            out[i] = toks[i][::-1]
    return " ".join(out)
