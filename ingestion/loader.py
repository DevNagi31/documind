"""Multi-format document loader: PDF, Markdown, plain text, and source code."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cpp", ".h"}
TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".rst"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = CODE_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class Document:
    content: str
    source: str
    doc_type: str  # "pdf" | "markdown" | "text" | "code"
    metadata: dict = field(default_factory=dict)


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError("PyMuPDF is required for PDF parsing. pip install pymupdf") from e

    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"[page {i + 1}]\n{text}")
    return "\n\n".join(pages)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _classify(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext == ".md" or ext == ".markdown":
        return "markdown"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return None


def load_file(path: str | Path) -> Document | None:
    path = Path(path)
    doc_type = _classify(path)
    if doc_type is None:
        return None
    content = _read_pdf(path) if doc_type == "pdf" else _read_text(path)
    return Document(
        content=content,
        source=str(path),
        doc_type=doc_type,
        metadata={"filename": path.name, "extension": path.suffix.lower()},
    )


def load_documents(root: str | Path, recursive: bool = True) -> list[Document]:
    """Load every supported file under `root`. Skips unsupported extensions silently."""
    root = Path(root)
    if root.is_file():
        doc = load_file(root)
        return [doc] if doc else []

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    pattern = "**/*" if recursive else "*"
    docs: list[Document] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        doc = load_file(path)
        if doc is not None:
            docs.append(doc)
    return docs


def iter_supported(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
