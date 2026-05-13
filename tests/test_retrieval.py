"""Retrieval tests — uses a fake embedder / store, no network."""
from __future__ import annotations

from dataclasses import dataclass

from documind.retrieval.retriever import Retriever
from documind.retrieval.vector_store import StoredChunk


@dataclass
class FakeEmbedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0]
            for word in t.lower().split():
                if "gil" in word or "thread" in word:
                    v[0] += 1
                if "multiprocess" in word:
                    v[1] += 1
                if "async" in word or "io" in word:
                    v[2] += 1
            out.append(v)
        return out

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def query(self, query_embedding, k=8, where=None):
        # Rank by negative dot product distance proxy.
        scored = []
        for c in self._chunks:
            dot = sum(a * b for a, b in zip(query_embedding, c.embedding))
            # Distance: smaller is better; we want highest-dot first → use -dot.
            scored.append((dot, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for dot, c in scored[:k]:
            c.distance = 1.0 - dot / 5.0  # arbitrary monotone transform
            out.append(c)
        return out


def _mk_chunk(cid, text, emb):
    return StoredChunk(chunk_id=cid, text=text, source=cid, metadata={"source": cid}, embedding=emb, distance=None)


def test_retriever_returns_topk_without_mmr():
    chunks = [
        _mk_chunk("a", "GIL thread thread", [3.0, 0.0, 0.0]),
        _mk_chunk("b", "multiprocessing multiprocessing", [0.0, 2.0, 0.0]),
        _mk_chunk("c", "async io", [0.0, 0.0, 2.0]),
    ]
    store = FakeStore(chunks)
    retriever = Retriever(store, FakeEmbedder())
    out = retriever.retrieve("GIL thread", k=2, fetch_k=3, use_mmr=False)
    assert len(out) == 2
    assert out[0].chunk_id == "a"


def test_retriever_mmr_promotes_diverse_chunks():
    chunks = [
        _mk_chunk("near1", "GIL thread", [3.0, 0.0, 0.0]),
        _mk_chunk("near2", "GIL thread thread", [3.0, 0.0, 0.0]),  # near-duplicate of near1
        _mk_chunk("far", "multiprocessing", [1.5, 1.5, 0.0]),
    ]
    store = FakeStore(chunks)
    retriever = Retriever(store, FakeEmbedder())
    out = retriever.retrieve("GIL thread", k=2, fetch_k=3, use_mmr=True, mmr_lambda=0.3)
    ids = [r.chunk_id for r in out]
    assert "far" in ids, "MMR should promote a diverse chunk over a near-duplicate"


def test_retrieve_zero_k_returns_empty():
    store = FakeStore([])
    out = Retriever(store, FakeEmbedder()).retrieve("anything", k=0)
    assert out == []
