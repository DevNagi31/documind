"""ChromaDB interface for persistent vector storage."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./.chroma")


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    source: str
    metadata: dict
    embedding: list[float] | None = None
    distance: float | None = None


class VectorStore:
    """Thin wrapper around ChromaDB for indexing and similarity search."""

    def __init__(self, collection: str, persist_dir: str = DEFAULT_PERSIST_DIR):
        import chromadb

        self.collection_name = collection
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def add(
        self,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
    ) -> None:
        if not chunk_ids:
            return
        self._collection.upsert(
            ids=list(chunk_ids),
            documents=list(texts),
            embeddings=[list(e) for e in embeddings],
            metadatas=[_sanitize_metadata(m) for m in metadatas],
        )

    def query(
        self,
        query_embedding: Sequence[float],
        k: int = 8,
        where: dict | None = None,
    ) -> list[StoredChunk]:
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        embs = result.get("embeddings", [[None] * len(ids)])[0]

        out: list[StoredChunk] = []
        for cid, text, meta, dist, emb in zip(ids, docs, metas, dists, embs):
            out.append(
                StoredChunk(
                    chunk_id=cid,
                    text=text,
                    source=(meta or {}).get("source", ""),
                    metadata=meta or {},
                    embedding=list(emb) if emb is not None else None,
                    distance=float(dist),
                )
            )
        return out

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]


def _sanitize_metadata(meta: dict) -> dict:
    """ChromaDB only accepts str/int/float/bool. Coerce or drop other types."""
    clean: dict = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
