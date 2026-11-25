"""Answer evaluation: is the generated answer on the right law, supported by its sources, and responsive?

A fixed sample of ``eval/questions.jsonl`` is run through the whole system (retrieval, reranking, Gemini) in
French and in Arabic. Each answer is then scored three ways:

- ``cites_expected``: one of the texts expected for the question is among the sources the answer cites
  (automatic, no model involved);
- ``grounded`` / ``responsive``: an LLM judge reads the question, the sources given to the generator and the
  answer, and says whether every legal statement is supported by the sources and whether the question is
  actually answered. The judge is asked for a different Gemini model than the one that usually writes the answer;
- a human (or another model) can read ``eval/results/answers.jsonl`` and fill in the last word.

There are no reference answers: the judge checks the answer against the retrieved text, not against the law
as a lawyer knows it. A faithful answer built on an outdated source still counts as grounded.

Both steps are resumable (the free Gemini tier is rate-limited).

Usage:
    python -m easylaw_rag.eval.answers generate [--n 30] [--langs fr ar]
    python -m easylaw_rag.eval.answers judge [--judge independent]   # independent: OPENROUTER_API_KEY in .env
    python -m easylaw_rag.eval.answers report
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

from easylaw_rag.config import DATA
from easylaw_rag.eval.retrieval import RESULTS, load_questions, resolve

ANSWERS = RESULTS / "answers.jsonl"
JUDGED = {"gemini": RESULTS / "answers_judged.jsonl", "independent": RESULTS / "answers_judged_independent.jsonl"}
SEED = 0
JUDGE_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
INDEPENDENT_JUDGE_MODELS = [m.strip() for m in os.getenv(
    "JUDGE_MODELS", "nvidia/nemotron-3-ultra-550b-a55b:free,baidu/ernie-5.5-flash-preview:free"
).split(",") if m.strip()]
PAUSE = 4.0  # seconds between Gemini calls, to stay under the free per-minute limit

JUDGE_SYSTEM = """Tu es un évaluateur rigoureux de réponses juridiques. On te donne une question, les SOURCES numérotées qui ont été fournies à un assistant, et la RÉPONSE de cet assistant (en français ou en arabe).

Évalue uniquement à partir des sources, pas de tes propres connaissances.

Réponds par un seul objet JSON, sans texte autour :
{
  "grounded": "oui" | "partiel" | "non",
  "responsive": "oui" | "partiel" | "non",
  "abstained": true | false,
  "problem": "..."
}

- grounded : "oui" si toutes les affirmations juridiques de la réponse (règles, durées, montants, âges, numéros d'articles) se trouvent dans les sources citées ; "partiel" si l'essentiel y est mais qu'un détail n'y figure pas ou est déformé ; "non" si une affirmation centrale est absente des sources ou les contredit.
- responsive : "oui" si la réponse répond directement et complètement à la question ; "partiel" si elle ne répond qu'à une partie ou reste vague ; "non" si elle ne répond pas.
- abstained : true si l'assistant déclare ne pas avoir trouvé la réponse dans les textes.
- problem : une phrase courte en français décrivant le défaut principal, ou "" s'il n'y en a pas."""


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample(n: int) -> list[dict]:
    questions = load_questions()
    picked = set(random.Random(SEED).sample(range(len(questions)), min(n, len(questions))))
    return [q for i, q in enumerate(questions) if i in picked]


def generate(n: int, langs: list[str]) -> None:
    from easylaw_rag.rag.answer import LegalRAG, build_prompt

    texts = pd.read_parquet(DATA / "interim" / "texts_fr.parquet", columns=["id_text", "type_text", "jt_number"])
    done = {(r["id"], r["lang"]) for r in read_jsonl(ANSWERS)}
    todo = [(q, lang) for q in sample(n) for lang in langs if (q["id"], lang) not in done]
    print(f"{len(done)} answers already saved, {len(todo)} to generate", flush=True)
    if not todo:
        return
    rag = LegalRAG()
    for i, (q, lang) in enumerate(todo, 1):
        expected = resolve(q["expected"], texts)
        try:
            ans = rag.ask(q[lang], lang=lang)
        except Exception as e:  # noqa: BLE001 - all models down: skip, a re-run picks it up
            print(f"  {i}/{len(todo)} {q['id']} {lang}: FAILED {e}", flush=True)
            time.sleep(30)
            continue
        cited = [s for s in ans.sources if s.n in ans.cited]
        append_jsonl(ANSWERS, {
            "id": q["id"], "lang": lang, "question": q[lang], "expected": q["expected"],
            "answer": ans.text.strip(), "model": ans.model, "seconds": round(ans.seconds, 1),
            "sources": [{"n": s.n, "citation": s.citation, "cited": s.n in ans.cited,
                         "expected": s.id_text in expected, "status": s.status} for s in ans.sources],
            "retrieved_expected": any(s.id_text in expected for s in ans.sources),
            "cites_expected": any(s.id_text in expected for s in cited),
            "context": build_prompt(ans.standalone, ans.sources, ans.lang),
        })  # fmt: skip
        print(f"  {i}/{len(todo)} {q['id']} {lang}: {ans.model}, {ans.seconds:.0f}s", flush=True)
        time.sleep(PAUSE)


def parse_verdict(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if v.get("grounded") in ("oui", "partiel", "non") and v.get("responsive") in ("oui", "partiel", "non"):
        return {"grounded": v["grounded"], "responsive": v["responsive"],
                "abstained": bool(v.get("abstained")), "problem": str(v.get("problem") or "")}  # fmt: skip
    return None


def gemini_judge():
    """(prompt -> (text, model)) on the Gemini free tier; no retriever, no GPU."""
    from google import genai
    from google.genai import types

    from easylaw_rag.config import google_api_key
    from easylaw_rag.rag.answer import LegalRAG

    rag = LegalRAG.__new__(LegalRAG)
    rag.models = JUDGE_MODELS
    rag.client = genai.Client(
        api_key=google_api_key(),
        http_options=types.HttpOptions(timeout=60000, retry_options=types.HttpRetryOptions(attempts=1)),
    )
    return lambda prompt: rag._generate(prompt, JUDGE_SYSTEM)


def independent_judge():
    """Same instructions, given to a model of another family through an OpenAI-compatible endpoint."""
    from easylaw_rag.rag.openai_compat import chat

    return lambda prompt: chat(JUDGE_SYSTEM, prompt, INDEPENDENT_JUDGE_MODELS)


def judge(which: str = "gemini") -> None:
    out = JUDGED[which]
    done = {(r["id"], r["lang"]) for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(ANSWERS) if (r["id"], r["lang"]) not in done]
    print(f"[{which}] {len(done)} answers already judged, {len(todo)} to judge", flush=True)
    if not todo:
        return
    call = gemini_judge() if which == "gemini" else independent_judge()
    for i, r in enumerate(todo, 1):
        prompt = f"{r['context']}\n\n=== RÉPONSE DE L'ASSISTANT ===\n{r['answer']}"
        verdict, model = None, "-"
        for _ in range(2):
            try:
                text, model = call(prompt)
            except Exception as e:  # noqa: BLE001
                print(f"  {i}/{len(todo)} {r['id']} {r['lang']}: judge failed ({e})", flush=True)
                if "per-day" in str(e):  # daily free quota reached: a re-run tomorrow resumes here
                    return
                time.sleep(30)
                continue
            verdict = parse_verdict(text)
            if verdict:
                break
            time.sleep(PAUSE)
        if verdict is None:
            continue
        append_jsonl(out, {"id": r["id"], "lang": r["lang"], "judge": model, **verdict})
        print(f"  {i}/{len(todo)} {r['id']} {r['lang']}: grounded {verdict['grounded']}, "
              f"responsive {verdict['responsive']} ({model})", flush=True)  # fmt: skip
        time.sleep(PAUSE)


def report() -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(ANSWERS)).drop(columns=["context", "sources"])
    judges = []
    for name, path in JUDGED.items():
        rows = read_jsonl(path)
        if rows:
            cols = ["judge", "grounded", "responsive", "abstained", "problem"]
            df = df.merge(pd.DataFrame(rows).rename(columns={c: f"{c}_{name}" for c in cols}), on=["id", "lang"], how="left")
            judges.append(name)
    rows = []
    for lang, g in df.groupby("lang"):
        row = {"lang": lang, "answers": len(g), "retrieved_expected": g["retrieved_expected"].mean(),
               "cites_expected": g["cites_expected"].mean(), "median_seconds": g["seconds"].median()}  # fmt: skip
        for name in judges:
            j = g.dropna(subset=[f"grounded_{name}"])
            ok = (j[f"grounded_{name}"] == "oui") & (j[f"responsive_{name}"] == "oui")
            row |= {f"judged_{name}": len(j), f"grounded_{name}": (j[f"grounded_{name}"] == "oui").mean(),
                    f"responsive_{name}": (j[f"responsive_{name}"] == "oui").mean(),
                    f"abstained_{name}": j[f"abstained_{name}"].astype(bool).mean(),
                    f"all_three_{name}": (ok & j["cites_expected"]).mean()}  # fmt: skip
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "answers_summary.csv", index=False)
    df.drop(columns=["answer"]).to_csv(RESULTS / "answers_detail.csv", index=False, encoding="utf-8-sig")
    print(summary.round(3).T.to_string(header=False))
    print("\nmodels:", df.groupby(["lang", "model"]).size().to_dict())
    return df


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["generate", "judge", "report"])
    ap.add_argument("--n", type=int, default=30, help="questions sampled (each asked in every language)")
    ap.add_argument("--langs", nargs="+", choices=["ar", "fr"], default=["fr", "ar"])
    ap.add_argument("--judge", choices=list(JUDGED), default="gemini", help="which judge to run")
    args = ap.parse_args()
    if args.step == "generate":
        generate(args.n, args.langs)
    elif args.step == "judge":
        judge(args.judge)
    else:
        report()


if __name__ == "__main__":
    main()
