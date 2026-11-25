"""Document metadata: Arabic citations, JORADP links, and amendment status."""

from __future__ import annotations

import re

import pandas as pd

from easylaw_rag.text.arabic import normalize

# Month names as used by the Algerian Journal Officiel (August = غشت).
MONTHS_AR = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "غشت", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]

# adjustment_type -> (field on the adjusted text, inverse field on the adjusting text)
RELATIONS = {
    "معدل بـ": ("amended_by", "amends"),
    "ملغى بـ": ("repealed_by", "repeals"),
    "نص تطبيقي": ("implemented_by", "implements"),
    "استدراك في": ("corrected_by", "corrects"),
    "موافق عليه بـ": ("approved_by", "approves"),
}
RELATION_FIELDS = [f for pair in RELATIONS.values() for f in pair]

# Individual or one-off measures: they state no general rule and, being numerous and near-identical,
# crowd real legislation out of the top results (247 naturalisation decrees outranked the nationality code).
_ADMIN_RE = re.compile(
    # careers; the verb is required: "ترقية الاستثمار" (investment promotion) is not a staff promotion
    r"(?:يتضمن|تتضمن|يتضمنان|المتضمن)\s+(?:ال)?(?:تعيين|ترقيه|ترسيم|ادراج)"
    r"|انهاء مهام|حركه في|حركه موظفين|احاله علي التقاعد"
    r"|تفويض الامضاء"  # signature delegations
    r"|(?:تحويل|نقل) اعتماد|توزيع الاعتمادات|احداث باب|يعدل توزيع نفقات"  # budget transfers
    r"|التجني?س ب|(?:اكتساب|منح) الجنسيه|استرداد الجنسيه|فقدان الجنسيه|تغيير(?: ال)?لقب|تغيير الالقاب"  # named persons
    r"|^(?:يتضمن|يتضمنان|تتضمن) منح (?:وسام|اوسمه)|القايمه الاسميه|^الوضعيه الشهريه"  # medals, name lists, bank statements
    r"|تخصيص قطعه|التنازل مجانا|تسميه القريه"  # single plots of land
)


def format_date_ar(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return f"{ts.day} {MONTHS_AR[ts.month - 1]} {ts.year}"


def citation(type_text: str, jt_number, signature_date, oj_number, oj_year, page) -> str:
    head = type_text
    if isinstance(jt_number, str) and jt_number.strip():
        head += f" رقم {jt_number.strip()}"
    if (d := format_date_ar(signature_date)) is not None:
        head += f" مؤرخ في {d}"
    if pd.notna(oj_number):
        head += f"، الجريدة الرسمية عدد {int(oj_number)} لسنة {int(oj_year)}"
        if pd.notna(page):
            head += f"، ص {int(page)}"
    return head


def joradp_pdf_url(oj_year, oj_number, lang: str = "ar") -> str | None:
    if pd.isna(oj_year) or pd.isna(oj_number):
        return None
    folder, letter = ("JO-ARABE", "A") if lang == "ar" else ("JO-FRANCAIS", "F")
    return f"http://www.joradp.dz/FTP/{folder}/{int(oj_year)}/{letter}{int(oj_year)}{int(oj_number):03d}.pdf"


def citation_fr(reference: str, oj_number, oj_year, page) -> str:
    """ "Décret exécutif n° 05-67 du 30 Janvier 2005, Journal officiel n° 10 de 2005, p. 4" """
    head = " ".join((reference or "").split())
    if pd.notna(oj_number) and pd.notna(oj_year):
        head += f", Journal officiel n° {int(oj_number)} de {int(oj_year)}"
        if pd.notna(page):
            head += f", p. {int(page)}"
    return head


# French counterpart of _ADMIN_RE (accents removed, lower case).
_ADMIN_FR_RE = re.compile(
    r"portant nomination|mettant fin aux fonctions|fin de fonctions|delegation de signature|mouvement dans"
    r"|titularisation|admission a la retraite|detachement|portant designation|chargeant .{0,40} de l'interim"
    r"|(?:transfert|virement|repartition) de(?:s)? credits?|creation d'un chapitre"
    r"|naturalisation|acquisition de la nationalite|reintegration dans la nationalite|changement de noms?"
    r"|attribution de (?:la |l')?(?:medaille|ordre)|liste nominative|situation mensuelle"
    r"|affectation d'une? (?:parcelle|terrain)|cession a titre gratuit|portant agrement d|retrait d'agrement"
)


def is_administrative_fr(description: str | None) -> bool:
    from easylaw_rag.text.lang import normalize_fr

    return bool(_ADMIN_FR_RE.search(normalize_fr(description or "")))


def is_administrative(description: str | None) -> bool:
    """Individual measures (appointments, promotions...) rather than general rules."""
    return bool(_ADMIN_RE.search(normalize(description or "")))


def relation_lists(adj: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
    """One row per text id with a list of related ids for every relation field."""
    out = pd.DataFrame(index=pd.Index(ids, name="id_text"))
    for kind, (forward, inverse) in RELATIONS.items():
        sub = adj[adj["adjustment_type"] == kind]
        fwd = sub.groupby("adjusted_num")["adjusting_num"].agg(lambda s: sorted(set(s)))
        inv = sub.groupby("adjusting_num")["adjusted_num"].agg(lambda s: sorted(set(s)))
        out[forward] = fwd.reindex(out.index)
        out[inverse] = inv.reindex(out.index)
    for col in RELATION_FIELDS:
        out[col] = out[col].map(lambda v: v if isinstance(v, list) else [])
    return out.reset_index()
