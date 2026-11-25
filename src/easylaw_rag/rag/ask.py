"""Ask a legal question from the command line.

Usage:
    python -m easylaw_rag.rag.ask "ما هي شروط الحصول على رخصة البناء؟"
    python -m easylaw_rag.rag.ask --in-force-only --show-context "..."
"""

from __future__ import annotations

import argparse
import sys

from easylaw_rag.rag.answer import LegalRAG
from easylaw_rag.retrieval.retriever import Filters, Retriever


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=8, help="chunks passed to the model")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: $EASYLAW_DEVICE or cuda)")
    ap.add_argument("--in-force-only", action="store_true")
    ap.add_argument("--include-administrative", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--show-context", action="store_true", help="print retrieved excerpts")
    args = ap.parse_args()

    rag = LegalRAG(Retriever(use_reranker=not args.no_rerank, device=args.device))
    filters = Filters(include_administrative=args.include_administrative, in_force_only=args.in_force_only)
    ans = rag.ask(args.question, k=args.k, filters=filters)

    print(ans.text.strip(), "\n")
    print("المصادر:")
    for s in ans.sources:
        mark = "●" if s.n in ans.cited else "○"
        print(f" {mark} [{s.n}] {s.citation}  (score {s.score:.2f})")
        print(f"       {s.description}")
        for line in s.status:
            print(f"       {line}")
        if s.pdf_url:
            print(f"       {s.pdf_url}")
        if args.show_context:
            for e in s.excerpts:
                print("       ┆ " + e[:400].replace("\n", " "))
    steps = ", ".join(f"{k} {v:.1f}s" for k, v in ans.timings.items())
    print(f"\n({ans.model}, {ans.seconds:.1f}s: {steps}; ● = cited in the answer)")


if __name__ == "__main__":
    main()
