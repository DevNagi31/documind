"""Prompt templates for RAG answer synthesis and LLM-as-judge evaluation."""
from __future__ import annotations

from typing import Sequence

RAG_SYSTEM = (
    "You are DocuMind, a precise knowledge assistant. Answer the user's question using ONLY "
    "the provided context. If the context does not contain enough information, say so honestly. "
    "Cite the relevant sources inline using the bracketed citation tags provided with each chunk. "
    "Do not invent facts."
)

RAG_PROMPT = """\
Context:
{context}

Question: {question}

Answer the question using only the context above. Cite sources with their bracketed tags like [source.md:12-30]. \
If the context is insufficient, reply: "I don't have enough information to answer that."\
"""


def build_rag_prompt(question: str, chunks: Sequence) -> tuple[str, str]:
    """Returns (system, user) prompts. `chunks` are RetrievedChunk-like with .text and .citation()."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        cite = c.citation() if hasattr(c, "citation") else f"[chunk-{i}]"
        blocks.append(f"{cite}\n{c.text.strip()}")
    context = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"
    return RAG_SYSTEM, RAG_PROMPT.format(context=context, question=question)


FAITHFULNESS_PROMPT = """\
You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Step 1: Extract every factual claim from the answer as a numbered list.
Step 2: For each claim, decide whether it is SUPPORTED by the context (yes/no).
Step 3: Output ONLY a JSON object on the final line: {{"supported": <int>, "total": <int>}}.\
"""

CONTEXT_PRECISION_PROMPT = """\
You are evaluating whether each retrieved chunk is relevant to answering the question.

Question: {question}

Chunks:
{chunks}

For each chunk, output "yes" if it contains information useful for answering the question, otherwise "no".
Output ONLY a JSON object on the final line: {{"verdicts": ["yes", "no", ...]}}\
"""

HALLUCINATION_PROMPT = """\
You are detecting hallucinations: statements in the answer that cannot be verified from the context.

Context:
{context}

Answer:
{answer}

List every statement in the answer that is NOT directly supported by the context.
Output ONLY a JSON object on the final line: {{"unsupported": <int>, "total": <int>}}.\
"""
