"""Groq client wrapper for fast Llama 3.3 70B inference (free tier)."""
from __future__ import annotations

import os
from typing import Sequence

from .prompts import build_rag_prompt

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com and put it in .env"
            )
        self.model = model
        self._client = None

    def _get(self):
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def chat(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        client = self._get()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def generate_answer(question: str, chunks: Sequence, client: GroqClient | None = None) -> str:
    client = client or GroqClient()
    system, user = build_rag_prompt(question, chunks)
    return client.chat(system, user)
