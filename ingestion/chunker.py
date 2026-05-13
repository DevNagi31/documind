"""Chunking strategies: recursive character splitting, code-aware splitting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .loader import Document

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80

CODE_SEPARATORS = ["\nclass ", "\ndef ", "\nfunction ", "\nif ", "\nfor ", "\nwhile ", "\n\n", "\n", " "]
TEXT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    start_line: int | None = None
    end_line: int | None = None
    metadata: dict = field(default_factory=dict)

    def chunk_id(self) -> str:
        return f"{self.source}::chunk-{self.chunk_index}"


def _split_recursive(text: str, separators: Sequence[str], chunk_size: int, overlap: int) -> list[str]:
    """Recursive character splitter: try larger separators first, fall back to smaller."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            chunks: list[str] = []
            buf = ""
            for part in parts:
                piece = part if not buf else buf + sep + part
                if len(piece) <= chunk_size:
                    buf = piece
                else:
                    if buf:
                        chunks.append(buf)
                    if len(part) > chunk_size:
                        chunks.extend(_split_recursive(part, separators[separators.index(sep) + 1 :], chunk_size, overlap))
                        buf = ""
                    else:
                        buf = part
            if buf:
                chunks.append(buf)
            return _apply_overlap(chunks, overlap)

    # Hard fall-through: fixed-size split.
    return _apply_overlap([text[i : i + chunk_size] for i in range(0, len(text), chunk_size)], overlap)


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) < 2:
        return [c for c in chunks if c.strip()]
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:] if len(chunks[i - 1]) > overlap else chunks[i - 1]
        result.append(prev_tail + chunks[i])
    return [c for c in result if c.strip()]


def _line_span(full_text: str, chunk_text: str, search_from: int) -> tuple[int, int, int]:
    """Return (start_line, end_line, next_search_from). Best-effort, used for citations."""
    idx = full_text.find(chunk_text, search_from)
    if idx < 0:
        return (None, None, search_from)
    start_line = full_text.count("\n", 0, idx) + 1
    end_line = start_line + chunk_text.count("\n")
    return (start_line, end_line, idx + len(chunk_text))


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    separators = CODE_SEPARATORS if doc.doc_type == "code" else TEXT_SEPARATORS
    pieces = _split_recursive(doc.content, separators, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    search_from = 0
    for i, piece in enumerate(pieces):
        start_line, end_line, search_from = _line_span(doc.content, piece, search_from)
        chunks.append(
            Chunk(
                text=piece,
                source=doc.source,
                chunk_index=i,
                start_line=start_line,
                end_line=end_line,
                metadata={**doc.metadata, "doc_type": doc.doc_type},
            )
        )
    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return out
