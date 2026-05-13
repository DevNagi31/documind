"""Context precision: fraction of retrieved chunks that are actually relevant."""
from __future__ import annotations

from typing import Sequence

from ..generation.llm import GroqClient
from ..generation.prompts import CONTEXT_PRECISION_PROMPT
from ._judge import parse_judge_json, safe_ratio


def score_context_precision(question: str, chunks: Sequence, client: GroqClient | None = None) -> float:
    if not chunks:
        return 0.0
    client = client or GroqClient()
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] {c.text.strip()}")
    prompt = CONTEXT_PRECISION_PROMPT.format(question=question, chunks="\n\n".join(blocks))
    response = client.chat(
        system="You are a strict evaluator. Be objective and respond in the requested format.",
        user=prompt,
        temperature=0.0,
    )
    payload = parse_judge_json(response)
    verdicts = payload.get("verdicts", [])
    if not verdicts:
        return 0.0
    yes = sum(1 for v in verdicts if str(v).strip().lower() == "yes")
    return safe_ratio(yes, len(verdicts))
