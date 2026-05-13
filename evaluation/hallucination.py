"""Hallucination rate: fraction of answer statements unsupported by context."""
from __future__ import annotations

from typing import Sequence

from ..generation.llm import GroqClient
from ..generation.prompts import HALLUCINATION_PROMPT
from ._judge import parse_judge_json, safe_ratio


def score_hallucination(answer: str, chunks: Sequence, client: GroqClient | None = None) -> float:
    """Returns a rate in [0, 1] — lower is better. 0.0 = fully grounded, 1.0 = fully hallucinated."""
    if not answer.strip() or not chunks:
        return 0.0
    client = client or GroqClient()
    context = "\n\n".join(c.text for c in chunks)
    prompt = HALLUCINATION_PROMPT.format(context=context, answer=answer)
    response = client.chat(
        system="You are a strict evaluator. Be objective and respond in the requested format.",
        user=prompt,
        temperature=0.0,
    )
    payload = parse_judge_json(response)
    return safe_ratio(payload.get("unsupported", 0), payload.get("total", 0))
