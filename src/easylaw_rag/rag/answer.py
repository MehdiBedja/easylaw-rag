"""Grounded answer generation: retrieved legal articles -> Gemini -> answer with numbered citations."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

import pandas as pd

from easylaw_rag.config import COLLECTIONS, corpus_dirs, google_api_key
from easylaw_rag.retrieval.retriever import Filters, Hit, Retriever, detect_lang

# Tried in order; on overload (503/504/timeout) the next one answers. Measured Nov 2026 on the free tier:
# 3.8/3.7-flash were saturated (503), 3.5-flash ~4s, 3.6-flash ~6s, 3.5-flash-lite ~1s (no reasoning).
# Either flash model can also hang for a minute. In the Nov 2026 answer evaluation 45/60 answers ended up
# coming from lite after flash failed, so the reliable lite model goes first and flash is only a fallback.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.6-flash"
).split(",") if m.strip()]
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "15000"))
MAX_RELATED = 3  # amending/repealing texts listed per source

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في التشريع الجزائري، تجيب اعتمادا على نصوص الجريدة الرسمية للجمهورية الجزائرية فقط.

قواعد صارمة:
1. أجب فقط من المصادر المرقمة المقدمة لك. لا تستعمل معلوماتك الخاصة ولا تخمّن.
2. أرفق كل معلومة برقم مصدرها بين معقوفين، مثل [1] أو [2][3].
3. إذا كان المصدر "ملغى" أو "معدّلا" فنبّه إلى ذلك صراحة، ووضّح أن الحكم قد لا يكون ساري المفعول بصيغته الحالية.
4. إذا تعارضت المصادر في نفس المسألة (مدتان أو سنّان أو مبلغان مختلفان)، فالنص الأحدث تاريخا هو الذي عدّل الأقدم في الغالب: قدّم الحكم الأحدث على أنه الساري، واذكر أن الحكم الأقدم قد عُدّل أو استُبدل، ولا تعرضهما كأنهما ساريان معا.
5. إذا لم تكف المصادر للإجابة، فقل ذلك بوضوح ("لم أجد في النصوص المتوفرة ما يجيب عن هذا السؤال")، واذكر ما وجدته من نصوص ذات صلة إن وجدت.
6. النصوص مستخرجة آليا (OCR) وقد تحتوي أخطاء في الأرقام والتواريخ والكلمات: اعتمد على سطر "المرجع" لأرقام النصوص وتواريخها، ولا تنقل أرقاما مشوهة.
7. أجب بلغة السؤال (العربية أو الفرنسية)، بأسلوب واضح ومنظم، دون إطالة.
8. هذه معلومات قانونية عامة وليست استشارة قانونية."""

SYSTEM_PROMPT_FR = """Tu es un assistant juridique spécialisé en droit algérien. Tu réponds uniquement à partir des textes du Journal officiel de la République algérienne qui te sont fournis.

Règles strictes :
1. Réponds uniquement à partir des sources numérotées fournies. N'utilise pas tes propres connaissances et ne devine pas.
2. Indique après chaque information le numéro de sa source entre crochets, par exemple [1] ou [2][3].
3. Si une source est « abrogée » ou « modifiée », signale-le explicitement et précise que la disposition peut ne plus être en vigueur dans cette rédaction.
4. Si des sources se contredisent sur un même point (deux durées, âges ou montants différents), le texte le plus récent a en général modifié le plus ancien : présente la règle la plus récente comme celle en vigueur et indique que l'ancienne a été modifiée ou remplacée, sans les présenter comme applicables en même temps.
5. Si les sources ne suffisent pas pour répondre, dis-le clairement (« Je n'ai pas trouvé dans les textes disponibles de réponse à cette question ») et mentionne les textes voisins trouvés, s'il y en a.
6. Certains textes anciens sont issus d'une reconnaissance optique (OCR) et peuvent contenir des erreurs : fie-toi à la ligne « Référence » pour les numéros et les dates des textes.
7. Réponds dans la langue de la question, de façon claire et structurée, sans longueur inutile.
8. Il s'agit d'une information juridique générale et non d'une consultation juridique."""

# Labels used in the prompt and in the status lines, per corpus language.
LABELS = {
    "ar": {
        "system": SYSTEM_PROMPT, "ref": "المرجع", "subject": "الموضوع", "status": "الحالة", "articles": "المواد",
        "text": "النص", "no_text": "(النص الكامل غير متوفر — يتوفر المرجع فقط)", "sources": "المصادر",
        "question": "السؤال", "sep": "، ", "repealed": "⚠️ ملغى (كليا أو جزئيا) بموجب", "amended": "معدّل بموجب",
        "none": "لم أجد في النصوص المتوفرة ما يجيب عن هذا السؤال.",
    },
    "fr": {
        "system": SYSTEM_PROMPT_FR, "ref": "Référence", "subject": "Objet", "status": "Statut", "articles": "Articles",
        "text": "Texte", "no_text": "(texte intégral non disponible — référence seule)", "sources": "Sources",
        "question": "Question", "sep": ", ", "repealed": "⚠️ Abrogé (en tout ou partie) par", "amended": "Modifié par",
        "none": "Je n'ai pas trouvé dans les textes disponibles de réponse à cette question.",
    },
}  # fmt: skip

# Follow-up questions ("and for part-time workers?") are rewritten into a standalone question before retrieval.
CONDENSE_PROMPT = """أعد صياغة السؤال الأخير ليكون سؤالا قانونيا مستقلا ومفهوما دون الحاجة إلى المحادثة السابقة، بنفس لغته.
إذا كان مستقلا أصلا فأعده كما هو. أجب بالسؤال المعاد صياغته فقط، دون أي شرح."""
MAX_HISTORY_TURNS = 3


@dataclass
class Source:
    n: int
    id_text: str
    citation: str
    description: str
    pdf_url: str | None
    status: list[str]
    articles: list[str]
    excerpts: list[str] = field(repr=False)
    score: float = 0.0


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source]
    cited: list[int]
    model: str
    seconds: float
    timings: dict[str, float] = field(default_factory=dict)
    standalone: str = ""  # the question actually searched (differs from ``question`` for follow-ups)
    lang: str = "ar"


@lru_cache(maxsize=2)
def _citations(lang: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for folder in corpus_dirs(lang):
        docs = pd.read_parquet(folder / "documents.parquet", columns=["id_text", "citation"])
        out.update(zip(docs["id_text"], docs["citation"]))
    return out


def status_lines(payload: dict, lang: str = "ar") -> list[str]:
    cites, lab = _citations(lang), LABELS[lang]

    def names(ids) -> str:
        ids = list(ids or [])
        shown = [cites.get(i, i) for i in ids[:MAX_RELATED]]
        more = f" (+{len(ids) - MAX_RELATED})" if len(ids) > MAX_RELATED else ""
        return " ; ".join(shown) + more

    lines = []
    if payload.get("repealed_by"):
        lines.append(f"{lab['repealed']}: {names(payload['repealed_by'])}")
    if payload.get("amended_by"):
        lines.append(f"{lab['amended']}: {names(payload['amended_by'])}")
    return lines


def group_sources(hits: list[Hit], lang: str = "ar") -> list[Source]:
    """One numbered source per legal text, keeping retrieval order; several chunks may share a source."""
    by_doc: dict[str, Source] = {}
    for h in hits:
        p = h.payload
        doc = p["id_text"]
        if doc not in by_doc:
            by_doc[doc] = Source(
                n=len(by_doc) + 1,
                id_text=doc,
                citation=p.get("citation") or doc,
                description=p.get("description") or "",
                pdf_url=p.get("pdf_url"),
                status=status_lines(p, lang),
                articles=[],
                excerpts=[],
                score=h.score,
            )
        src = by_doc[doc]
        src.articles += [a for a in (p.get("articles") or []) if a not in src.articles]
        if h.text:
            src.excerpts.append(h.text)
    return list(by_doc.values())


def build_prompt(question: str, sources: list[Source], lang: str = "ar") -> str:
    lab = LABELS[lang]
    blocks = []
    for s in sources:
        lines = [f"[{s.n}] {lab['ref']}: {s.citation}", f"{lab['subject']}: {s.description}"]
        lines += [f"{lab['status']}: {line}" for line in s.status]
        if s.articles:
            lines.append(f"{lab['articles']}: " + lab["sep"].join(s.articles))
        lines.append(f"{lab['text']}:\n" + ("\n…\n".join(s.excerpts) or lab["no_text"]))
        blocks.append("\n".join(lines))
    return f"{lab['sources']}:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\n{lab['question']}: {question}"


class LegalRAG:
    def __init__(self, retriever: Retriever | None = None, models: list[str] | None = None):
        from google import genai
        from google.genai import types

        self.retriever = retriever or Retriever()
        self.models = models or GEMINI_MODELS
        # SDK-level retries disabled: on an overloaded model we fail fast and fall back instead of waiting minutes.
        self.client = genai.Client(
            api_key=google_api_key(),
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)),
        )

    def _generate(
        self, prompt: str, system: str = SYSTEM_PROMPT, models: list[str] | None = None, rate_limit_waits: int = 3
    ) -> tuple[str, str]:
        """Return (answer text, model used)."""
        from google.genai import errors, types

        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )
        failures = []
        for model in models or self.models:
            for _ in range(rate_limit_waits + 1):
                try:
                    r = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
                    return r.text or "", model
                except errors.ClientError as e:
                    if e.code != 429:
                        raise
                    m = re.search(r"retry in ([\d.]+)s", str(e))  # free-tier per-minute limit: wait, retry
                    time.sleep(min(60.0, float(m.group(1)) + 1) if m else 10)
                    failures.append(f"{model}: 429")
                except (errors.ServerError, TimeoutError, OSError) as e:  # overloaded / timeout: next model
                    failures.append(f"{model}: {getattr(e, 'code', type(e).__name__)}")
                    break
                except Exception as e:  # httpx timeouts surface as their own types
                    if "timeout" not in type(e).__name__.lower():
                        raise
                    failures.append(f"{model}: timeout")
                    break
        raise RuntimeError("All Gemini models failed: " + "; ".join(failures))

    def condense(self, question: str, history: list[tuple[str, str]]) -> str:
        """Standalone version of a follow-up question; ``history`` is [(question, answer), ...], oldest first."""
        turns = "\n\n".join(f"س: {q}\nج: {a[:600]}" for q, a in history[-MAX_HISTORY_TURNS:])
        try:  # fastest model first: this is a rewrite, not legal reasoning
            text, _ = self._generate(f"{turns}\n\nالسؤال الأخير: {question}", CONDENSE_PROMPT, sorted(self.models, key=lambda m: "lite" not in m), 0)
        except Exception:  # noqa: BLE001 - a failed rewrite must never block the answer
            return question
        return text.strip() or question

    def ask(
        self, question: str, k: int = 8, filters: Filters | None = None, history: list[tuple[str, str]] | None = None,
        lang: str | None = None,
    ) -> Answer:
        """``lang`` selects the corpus ("ar" / "fr"); by default it follows the language of the question."""
        t0 = time.perf_counter()
        asked, question = question, self.condense(question, history) if history else question
        lang = lang or detect_lang(question)
        if not self.retriever.qc.collection_exists(COLLECTIONS[lang]):
            lang = "ar"
        lab = LABELS[lang]
        hits = self.retriever.search(question, k=k, filters=filters, collection=COLLECTIONS[lang])
        sources = group_sources(hits, lang)
        t1 = time.perf_counter()
        model = "-"
        if not sources:
            text = lab["none"]
        else:
            text, model = self._generate(build_prompt(question, sources, lang), lab["system"])
        t2 = time.perf_counter()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(sources)})
        timings = {**self.retriever.last_timings, "generate": t2 - t1}
        return Answer(asked, text, sources, cited, model, t2 - t0, timings, standalone=question, lang=lang)
