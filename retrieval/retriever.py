"""Retrieval with similarity search and MMR (Maximal Marginal Relevance) diversity."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..ingestion.embedder import Embedder
from .vector_store import StoredChunk, VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    metadata: dict
    score: float  # similarity in [0, 1]; higher = more relevant

    def citation(self) -> str:
        meta = self.metadata or {}
        start = meta.get("start_line")
        end = meta.get("end_line")
        loc = f":{start}-{end}" if start and end else ""
        return f"[{meta.get('filename', self.source)}{loc}]"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _to_retrieved(c: StoredChunk) -> RetrievedChunk:
    # ChromaDB cosine returns a distance in [0, 2]; convert to similarity in [0, 1].
    sim = 1.0 - (c.distance or 0.0) / 2.0
    return RetrievedChunk(
        chunk_id=c.chunk_id,
        text=c.text,
        source=c.source,
        metadata=c.metadata,
        score=max(0.0, min(1.0, sim)),
    )


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        if k <= 0:
            return []
        query_vec = self.embedder.embed_one(query)
        candidates = self.store.query(query_vec, k=max(fetch_k, k), where=where)
        if not candidates:
            return []
        if not use_mmr:
            return [_to_retrieved(c) for c in candidates[:k]]
        return self._mmr(query_vec, candidates, k=k, lambda_=mmr_lambda)

    def _mmr(
        self,
        query_vec: list[float],
        candidates: list[StoredChunk],
        k: int,
        lambda_: float,
    ) -> list[RetrievedChunk]:
        usable = [c for c in candidates if c.embedding is not None]
        if not usable:
            return [_to_retrieved(c) for c in candidates[:k]]

        sim_to_query = {c.chunk_id: _cosine(query_vec, c.embedding) for c in usable}  # type: ignore[arg-type]
        selected: list[StoredChunk] = []
        remaining = list(usable)

        while remaining and len(selected) < k:
            best, best_score = None, -math.inf
            for cand in remaining:
                redundancy = max(
                    (_cosine(cand.embedding, s.embedding) for s in selected),  # type: ignore[arg-type]
                    default=0.0,
                )
                score = lambda_ * sim_to_query[cand.chunk_id] - (1 - lambda_) * redundancy
                if score > best_score:
                    best, best_score = cand, score
            assert best is not None
            selected.append(best)
            remaining.remove(best)

        return [_to_retrieved(c) for c in selected]
