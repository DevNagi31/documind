"""Run the full evaluation suite over a list of test questions."""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..generation.llm import GroqClient, generate_answer
from ..ingestion.embedder import Embedder
from ..retrieval.retriever import Retriever
from .faithfulness import score_faithfulness
from .hallucination import score_hallucination
from .precision import score_context_precision
from .relevance import score_answer_relevance


@dataclass
class EvalResult:
    question: str
    answer: str
    sources: list[str]
    faithfulness: float
    answer_relevance: float
    context_precision: float
    hallucination: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std}


def load_test_set(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise ValueError("Test set must be a JSON list or {questions: [...]}")
    return data


def run_benchmark(
    retriever: Retriever,
    test_set: list[dict],
    k: int = 5,
    llm: GroqClient | None = None,
    embedder: Embedder | None = None,
    on_progress=None,
) -> tuple[list[EvalResult], dict]:
    """Returns (per-question results, aggregate summary)."""
    llm = llm or GroqClient()
    embedder = embedder or retriever.embedder

    results: list[EvalResult] = []
    for i, item in enumerate(test_set):
        question = item["question"]
        chunks = retriever.retrieve(question, k=k)
        answer = generate_answer(question, chunks, client=llm) if chunks else "I don't have enough information to answer that."

        faith = score_faithfulness(answer, chunks, client=llm) if chunks else 0.0
        rel = score_answer_relevance(question, answer, embedder=embedder)
        prec = score_context_precision(question, chunks, client=llm) if chunks else 0.0
        hall = score_hallucination(answer, chunks, client=llm) if chunks else 0.0

        results.append(
            EvalResult(
                question=question,
                answer=answer,
                sources=[c.citation() for c in chunks],
                faithfulness=faith,
                answer_relevance=rel,
                context_precision=prec,
                hallucination=hall,
                metadata={"id": item.get("id"), "expected": item.get("expected_answer")},
            )
        )
        if on_progress:
            on_progress(i + 1, len(test_set), results[-1])

    summary = {
        "n": len(results),
        "faithfulness": _summarize([r.faithfulness for r in results]),
        "answer_relevance": _summarize([r.answer_relevance for r in results]),
        "context_precision": _summarize([r.context_precision for r in results]),
        "hallucination": _summarize([r.hallucination for r in results]),
    }
    return results, summary
