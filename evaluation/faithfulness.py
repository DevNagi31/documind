"""Faithfulness: fraction of answer claims that are supported by retrieved context."""
from __future__ import annotations

from typing import Sequence

from ..generation.llm import GroqClient
from ..generation.prompts import FAITHFULNESS_PROMPT
from ._judge import parse_judge_json, safe_ratio


def score_faithfulness(answer: str, chunks: Sequence, client: GroqClient | None = None) -> float:
    if not answer.strip() or not chunks:
        return 0.0
    client = client or GroqClient()
    context = "\n\n".join(c.text for c in chunks)
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    response = client.chat(
        system="You are a strict evaluator. Be objective and respond in the requested format.",
        user=prompt,
        temperature=0.0,
    )
    payload = parse_judge_json(response)
    return safe_ratio(payload.get("supported", 0), payload.get("total", 0))
