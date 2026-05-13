"""CLI: run the evaluation suite against a test set.

    python -m documind.evaluate --collection my_docs --test-set eval_questions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .evaluation.benchmark import load_test_set, run_benchmark
from .generation.llm import GroqClient
from .ingestion.embedder import Embedder
from .retrieval.retriever import Retriever
from .retrieval.vector_store import VectorStore


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DocuMind evaluation suite.")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--test-set", required=True, help="Path to JSON file with questions")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--output", default=None, help="Write per-question results as JSON")
    args = parser.parse_args(argv)

    test_set = load_test_set(args.test_set)
    print(f"Loaded {len(test_set)} test questions", file=sys.stderr)

    store_kwargs = {"collection": args.collection}
    if args.persist_dir:
        store_kwargs["persist_dir"] = args.persist_dir
    store = VectorStore(**store_kwargs)
    embedder = Embedder()
    retriever = Retriever(store, embedder)
    llm = GroqClient()

    def on_progress(i: int, n: int, _result):
        print(f"  [{i}/{n}] done", file=sys.stderr)

    results, summary = run_benchmark(retriever, test_set, k=args.k, llm=llm, embedder=embedder, on_progress=on_progress)

    print("\nEvaluation Results")
    print("─" * 50)
    for metric in ("faithfulness", "answer_relevance", "context_precision", "hallucination"):
        s = summary[metric]
        bar = _bar(s["mean"] if metric != "hallucination" else 1 - s["mean"])
        print(f"  {metric:<20} {s['mean']:.2f} (±{s['std']:.2f}) {bar}")
    print(f"\n  n = {summary['n']} questions")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(
                {"summary": summary, "results": [r.to_dict() for r in results]},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote results to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
