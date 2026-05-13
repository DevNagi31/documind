"""Evaluation parsing/scoring tests — no LLM, no network."""
from __future__ import annotations

from dataclasses import dataclass

from documind.evaluation._judge import parse_judge_json, safe_ratio


@dataclass
class FakeLLM:
    response: str

    def chat(self, system, user, temperature=0.0, max_tokens=1024):
        return self.response


@dataclass
class _FakeChunk:
    text: str

    def citation(self):
        return "[x]"


def test_parse_judge_json_trailing_object():
    text = "Step 1...\nStep 2...\n{\"supported\": 3, \"total\": 4}"
    assert parse_judge_json(text) == {"supported": 3, "total": 4}


def test_parse_judge_json_fenced():
    text = "Here you go:\n```json\n{\"verdicts\": [\"yes\", \"no\"]}\n```"
    assert parse_judge_json(text) == {"verdicts": ["yes", "no"]}


def test_parse_judge_json_garbage_returns_empty():
    assert parse_judge_json("not parseable") == {}
    assert parse_judge_json("") == {}


def test_safe_ratio_bounds():
    assert safe_ratio(3, 4) == 0.75
    assert safe_ratio(0, 0) == 0.0
    assert safe_ratio(5, 4) == 1.0


def test_faithfulness_uses_llm_verdict(monkeypatch):
    from documind.evaluation import faithfulness as mod

    fake = FakeLLM(response='Step 1...\n{"supported": 4, "total": 5}')
    score = mod.score_faithfulness("answer", [_FakeChunk("ctx")], client=fake)
    assert score == 0.8


def test_precision_yes_no_counting(monkeypatch):
    from documind.evaluation import precision as mod

    fake = FakeLLM(response='{"verdicts": ["yes", "no", "yes", "yes"]}')
    score = mod.score_context_precision("q?", [_FakeChunk("a"), _FakeChunk("b"), _FakeChunk("c"), _FakeChunk("d")], client=fake)
    assert score == 0.75


def test_hallucination_rate(monkeypatch):
    from documind.evaluation import hallucination as mod

    fake = FakeLLM(response='{"unsupported": 1, "total": 4}')
    score = mod.score_hallucination("answer", [_FakeChunk("ctx")], client=fake)
    assert score == 0.25
