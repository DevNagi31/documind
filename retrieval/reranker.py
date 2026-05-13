"""Optional cross-encoder reranker. Improves precision at the cost of latency."""
from __future__ import annotations

from .retriever import RetrievedChunk

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        model = self._load()
        pairs = [(query, c.text) for c in chunks]
        scores = model.predict(pairs).tolist()
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        out = []
        for chunk, score in ranked:
            chunk.score = float(score)
            out.append(chunk)
        return out[:top_k] if top_k else out
