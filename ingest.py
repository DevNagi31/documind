"""CLI: ingest documents into a ChromaDB collection.

    python -m documind.ingest --path ./docs --collection my_docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .ingestion.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from .ingestion.embedder import Embedder
from .ingestion.loader import load_documents
from .retrieval.vector_store import VectorStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into DocuMind.")
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    parser.add_argument("--collection", required=True, help="Target ChromaDB collection name")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--persist-dir", default=None, help="ChromaDB persistence directory")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    root = Path(args.path)
    print(f"[1/4] Loading documents from {root}", file=sys.stderr)
    docs = load_documents(root)
    if not docs:
        print(f"No supported documents found at {root}", file=sys.stderr)
        return 1
    print(f"      Loaded {len(docs)} document(s)", file=sys.stderr)

    print(f"[2/4] Chunking (size={args.chunk_size}, overlap={args.chunk_overlap})", file=sys.stderr)
    chunks = chunk_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"      Produced {len(chunks)} chunks", file=sys.stderr)

    print(f"[3/4] Embedding chunks", file=sys.stderr)
    embedder = Embedder(args.embedding_model) if args.embedding_model else Embedder()
    vectors = embedder.embed(
        [c.text for c in chunks],
        batch_size=args.batch_size,
        show_progress=True,
    )

    print(f"[4/4] Writing to ChromaDB collection '{args.collection}'", file=sys.stderr)
    store_kwargs = {"collection": args.collection}
    if args.persist_dir:
        store_kwargs["persist_dir"] = args.persist_dir
    store = VectorStore(**store_kwargs)
    store.add(
        chunk_ids=[c.chunk_id() for c in chunks],
        texts=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[
            {
                **c.metadata,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ],
    )
    print(f"      Collection now contains {store.count()} chunks", file=sys.stderr)
    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
