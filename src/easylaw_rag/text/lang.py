"""Language profiles: everything the cleaning / segmentation / chunking code needs to know about a language.

The Journal Officiel is published in Arabic and French with the same structure (running headers, text
headers such as "Décret exécutif n° 05-67 du ..." / "مرسوم تنفيذي رقم 05-67 مؤرخ في ...", visas, articles),
so the pipeline is shared and only these patterns differ.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from easylaw_rag.text.arabic import ARABIC_WORD_RE, arabic_words, normalize


@dataclass(frozen=True)
class Lang:
    code: str
    normalize: Callable[[str], str]
    words: Callable[[str], list[str]]  # normalized content words, used to match titles
    word_re: re.Pattern  # a real word (on normalized text); lines without one are noise when short
    running_header_re: re.Pattern  # on normalized text
    date_banner_re: re.Pattern  # on normalized text
    article_re: re.Pattern  # on normalized text, group "num"
    break_before_re: re.Pattern  # on normalized text: openings that always start a paragraph
    header_re: re.Pattern  # start of a text header
    title_verb_re: re.Pattern  # on normalized text: verb introducing the object of a title
    authority_re: re.Pattern  # on normalized text: issuing-authority heading above a text
    ordinals: dict[str, str] = field(default_factory=dict)
    header_on_display: bool = False  # match header_re on the original-case line instead of the normalized one
    running_header_max_words: int = 12
    noise_max_len: int = 12  # lines shorter than this without a real word are dropped

    def is_header(self, line) -> bool:
        return bool(self.header_re.match(line.text if self.header_on_display else line.norm))


# ----------------------------------------------------------------------------------------------- Arabic

_AR_DAYS = r"(?:الاحد|الاثنين|الثلاثاء|الاربعاء|الخميس|الجمعه|السبت)"
_AR_TYPE_WORDS = (
    r"(?:مرسوم|مرسومان|مراسيم|قرار|قراران|قرارات|امر|امران|قانون|مقرر|مقرران|مقررات|منشور|راي|نظام"
    r"|اعلان|تعليمه|اتفاقيه|مداوله|تصريح)"
)

AR = Lang(
    code="ar",
    normalize=normalize,
    words=arabic_words,
    word_re=ARABIC_WORD_RE,
    running_header_re=re.compile(r"الجريده الرسميه|للجمهوريه الجزايريه الديمقراطيه"),
    date_banner_re=re.compile(rf"^\W*{_AR_DAYS}\b.*\bعام\s*\d{{3,4}}"),
    # A new paragraph is forced before these openings (OCR often glues them to the previous line).
    article_re=re.compile(
        r"^[-–\s]*(?:ال)?(?:ماده|ملاه|مااده)\s*(?P<num>\d+|؟|الاولي|الاول|الوحيده|[^\s:]{2,10}ه)?\s*[:؛.\-–]?"
    ),
    break_before_re=re.compile(
        r"^[-–\s]*(?:و\s*)?(?:بمقتضي|بناء|بعد|بموجب)\b|^[-–]\s|^(?:الباب|الفصل|القسم|العنوان)\s"
    ),
    # "<type> [up to 4 words] رقم|مؤرخ ..." at the start of a line, e.g. "مرسوم تنفيذي رقم 21-55 مؤرخ في".
    header_re=re.compile(rf"^[-–\s]*{_AR_TYPE_WORDS}(?:\s+\S+){{0,4}}?\s+(?:رقم|مورخ|مورخه|مورخات|مورخين)\b"),
    # Titles continue with a verb describing the object: "... يتضمن / يحدد / يعدل / يتعلق ...".
    title_verb_re=re.compile(
        r"\b(?:يتضمن|تتضمن|يتضمنان|يحدد|تحدد|يحددان|يعدل|يعدلان|يتمم|يتعلق|تتعلق|يتعلقان|يحدث|يرخص|ياذن"
        r"|يوافق|يلغي|ينشي|ينظم|يصادق|يجعل|يمنح|يعلن|يخص|يقرر|بشان"
        # 1960s style: "مرسوم ... بتغيير / بتحديد / بانشاء ..."
        r"|ب(?:تغيير|تعديل|تتميم|تحديد|انشاء|احداث|تعيين|تنظيم|تطبيق|الغاء|تاسيس|منح|اعلان))\b"
    ),
    # "وزارة المالية", "رئاسة الجمهورية", "مصالح الوزير الأول"
    authority_re=re.compile(r"^[-–\s]*(?:وزاره|رياسه|مصالح|المجلس|الامانه العامه|المحكمه|البنك)\b"),
    ordinals={"الاولي": "1", "الاول": "1", "الوحيده": "1"},
)

# ----------------------------------------------------------------------------------------------- French

_FR_STOP = frozenset(
    "les des une aux par pour sur dans avec sans sous que qui est ses son leur leurs cette ces "
    "ainsi dont entre vers chez elle ils elles nous vous ont ete etre comme mais donc car pas plus "
    "au du de la le et en un ou ne se ce sa".split()
)
_FR_WORD_RE = re.compile(r"[a-z]{3,}")


def normalize_fr(text: str) -> str:
    """Lowercase, accents and ligatures removed, typographic apostrophes and spaces unified."""
    text = unicodedata.normalize("NFKC", text).replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("‘", "'").replace("º", "°").lower()
    return re.sub(r"\s+", " ", text).strip()


def french_words(text: str) -> list[str]:
    return [w for w in _FR_WORD_RE.findall(normalize_fr(text)) if w not in _FR_STOP]


_FR_TYPES = (
    r"(?:Lois?|Ordonnances?|D[ée]crets?|Arr[êe]t[ée]s?|D[ée]cisions?|Instructions?|Circulaires?|Avis"
    r"|R[èe]glements?|D[ée]lib[ée]rations?|Proclamations?|R[ée]solutions?|D[ée]clarations?|Conventions?)"
)
FR = Lang(
    code="fr",
    normalize=normalize_fr,
    words=french_words,
    word_re=_FR_WORD_RE,
    # The page banner carries the issue or page number; the same words inside a sentence ("... publié au
    # Journal officiel de la République algérienne démocratique et populaire") do not.
    running_header_re=re.compile(r"^(?=.*\d).*journal officiel de la republique algerienne(?! democratique)"),
    # no date-banner rule: a date alone on a line is usually the wrapped end of "Fait à Alger, le ..."
    date_banner_re=re.compile(r"(?!)"),
    # "Article 1er. — ...", "Art. 2. — ...", "Article unique. — ..."; punctuation is required so that a
    # wrapped line "article 5 de la loi ..." is not taken for a new article.
    article_re=re.compile(
        r"^[-–—\s]*art(?:icle)?\s*\.?\s*(?P<num>\d+|1\s?er|ler|1°|premier|unique)\s*(?:bis|ter|quater)?\s*[.:\-–—]"
    ),
    break_before_re=re.compile(
        r"^(?:vu|sur le rapport|sur proposition|considerant|apres)\b|^[-–—]\s"
        r"|^(?:chapitre|titre|section|livre|annexe)\s"
    ),
    # Headers start with a capital ("Décret exécutif n° ... du ..."); visas quote texts in lower case
    # ("Vu le décret exécutif n° ..."), which is what keeps a wrapped visa from looking like a header.
    # A section heading printed on the same line ("LOIS Loi n° ...", "D E C R E T S Décret ...") is skipped.
    header_re=re.compile(
        rf"^[-–—\s]*(?:(?:[A-ZÉÈÊ]\s?){{3,}}\s+)?{_FR_TYPES}\b(?:\s+\S+){{0,5}}?\s+(?:n\s?[°ºo]|du|des)\s*\S"
    ),
    title_verb_re=re.compile(
        r"\b(?:portant|relatif|relative|relatifs|relatives|fixant|modifiant|completant|determinant|definissant"
        r"|instituant|autorisant|mettant|chargeant|rapportant|abrogeant|prorogeant|rendant|approuvant|declarant"
        r"|precisant|organisant|creant|accordant|ouvrant|conferant|habilitant|designant|reglementant|concernant"
        r"|tendant|supprimant|etablissant|erigeant|transferant|reportant|interdisant|soumettant|attribuant"
        r"|affectant|placant|convoquant|suspendant|homologuant|constatant|delimitant|classant|renouvelant"
        r"|retirant|adoptant|annulant|deleguant|dissolvant|integrant|instaurant|prescrivant|ratifiant)\b"
    ),
    authority_re=re.compile(
        r"^[-–—\s]*(?:ministere|presidence|services du|premier ministre|chef du gouvernement|conseil|cour"
        r"|banque|secretariat|autorite|haut|haute)\b"
    ),
    ordinals={"1er": "1", "1 er": "1", "ler": "1", "1°": "1", "premier": "1", "unique": "1"},
    header_on_display=True,
    running_header_max_words=20,
    noise_max_len=4,
)

LANGS = {"ar": AR, "fr": FR}
