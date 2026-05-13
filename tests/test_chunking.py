"""Chunking smoke tests — no network, no model downloads."""
from __future__ import annotations

from documind.ingestion.chunker import chunk_document, chunk_documents
from documind.ingestion.loader import Document


def test_small_doc_returns_single_chunk():
    doc = Document(content="Hello world.", source="a.txt", doc_type="text")
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].source == "a.txt"


def test_large_doc_is_split():
    paragraphs = ["This is a sentence." * 20] * 10
    doc = Document(content="\n\n".join(paragraphs), source="big.txt", doc_type="text")
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=30)
    assert len(chunks) > 1
    assert all(c.text.strip() for c in chunks)
    assert {c.chunk_index for c in chunks} == set(range(len(chunks)))


def test_code_uses_code_separators():
    code = (
        "import os\n\n"
        "def foo():\n    return 1\n\n"
        "def bar():\n    return 2\n\n"
        "class Baz:\n    pass\n"
    )
    doc = Document(content=code, source="m.py", doc_type="code")
    chunks = chunk_document(doc, chunk_size=40, chunk_overlap=0)
    assert len(chunks) >= 2


def test_chunk_id_is_deterministic():
    doc = Document(content="abc def ghi", source="x.txt", doc_type="text")
    chunks = chunk_document(doc)
    assert chunks[0].chunk_id() == "x.txt::chunk-0"


def test_chunk_documents_preserves_order():
    docs = [
        Document(content="alpha", source="1.txt", doc_type="text"),
        Document(content="beta", source="2.txt", doc_type="text"),
    ]
    chunks = chunk_documents(docs)
    assert [c.source for c in chunks] == ["1.txt", "2.txt"]
