"""PDF text extraction and overlapping chunking for TechManualRAG.

The module is intentionally dependency-light: page extraction relies on
PyMuPDF (imported as ``fitz``), while the chunking logic is pure standard
library so it can be unit-tested and reused without pulling in the heavy RAG
stack (LlamaIndex, Streamlit, Ollama, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 50


@dataclass(frozen=True)
class Page:
    """A single extracted PDF page.

    ``page_no`` is 1-based to match how humans reference document pages.
    """

    page_no: int
    text: str


@dataclass(frozen=True)
class PageChunks:
    """Overlapping text chunks belonging to one source page."""

    page_no: int
    chunks: list[str]


def extract_pages(pdf_bytes: bytes) -> list[Page]:
    """Extract per-page text from an in-memory PDF.

    Args:
        pdf_bytes: Raw bytes of a PDF document.

    Returns:
        One :class:`Page` per document page, with surrounding whitespace
        stripped. Pages without extractable text yield an empty ``text``.

    Raises:
        ValueError: If ``pdf_bytes`` is empty or is not a readable PDF.
    """
    if not pdf_bytes:
        raise ValueError("PDF input is empty (0 bytes).")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several low-level error types.
        raise ValueError(f"Could not open PDF: {exc}") from exc

    try:
        return [
            Page(page_no=index + 1, text=page.get_text().strip())
            for index, page in enumerate(doc)
        ]
    finally:
        doc.close()


def chunk_text(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping fixed-size character windows.

    Consecutive chunks share their last/first ``overlap`` characters, so no
    context is lost across a boundary. The final chunk may be shorter than
    ``size``.

    Args:
        text: The text to split.
        size: Maximum characters per chunk (must be positive).
        overlap: Characters shared between neighbours (0 <= overlap < size).

    Returns:
        The list of chunks; empty when ``text`` is empty.

    Raises:
        ValueError: If ``size``/``overlap`` are outside their valid ranges.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size.")

    if not text:
        return []

    step = size - overlap
    length = len(text)
    chunks: list[str] = []
    start = 0
    while start < length:
        end = start + size
        chunks.append(text[start:end])
        if end >= length:
            break
        start += step
    return chunks


def chunk_pages(
    pages: list[Page],
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[PageChunks]:
    """Chunk every page while preserving its 1-based page number."""
    return [
        PageChunks(page_no=page.page_no, chunks=chunk_text(page.text, size, overlap))
        for page in pages
    ]
