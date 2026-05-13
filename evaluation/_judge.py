"""Shared helpers for LLM-as-judge evaluators."""
from __future__ import annotations

import json
import re
from typing import Any


def parse_judge_json(text: str) -> dict[str, Any]:
    """Extract a trailing JSON object from a judge's response. Returns {} on failure."""
    if not text:
        return {}
    # Try a fenced code block first.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Otherwise take the last balanced JSON object on the page.
    matches = list(re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


def safe_ratio(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, numer / denom))
