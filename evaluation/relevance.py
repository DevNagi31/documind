"""Answer relevance: embedding cosine similarity between question and answer."""
from __future__ import annotations

import math

from ..ingestion.embedder import Embedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def score_answer_relevance(question: str, answer: str, embedder: Embedder | None = None) -> float:
    if not question.strip() or not answer.strip():
        return 0.0
    embedder = embedder or Embedder()
    q_vec, a_vec = embedder.embed([question, answer])
    sim = _cosine(q_vec, a_vec)
    # Normalize cosine ∈ [-1, 1] to a score ∈ [0, 1].
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))
