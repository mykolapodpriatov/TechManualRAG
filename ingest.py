"""PDF text extraction and overlapping chunking for TechManualRAG.

The module is intentionally dependency-light: page extraction relies on
PyMuPDF (imported as ``fitz``), while the chunking logic is pure standard
library so it can be unit-tested and reused without pulling in the heavy RAG
stack (LlamaIndex, Streamlit, Ollama, ...).

It also doubles as an offline CLI::

    python ingest.py manual.pdf --json --pages 1-3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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


def chunk_text_words(
    text: str,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping chunks without cutting words.

    Unlike :func:`chunk_text`, which slices on raw character offsets and can
    therefore end a chunk mid-word, this packs whole whitespace-delimited
    tokens into each chunk up to ``size`` characters. Consecutive chunks share
    a suffix/prefix of whole words spanning at least ``overlap`` characters (as
    far as whole words allow), so no token is ever split across a boundary.

    Args:
        text: The text to split. Runs of whitespace are normalised to single
            spaces within a chunk.
        size: Soft maximum characters per chunk (must be positive). A single
            token longer than ``size`` becomes its own oversized chunk rather
            than being cut.
        overlap: Approximate characters shared between neighbours
            (0 <= overlap < size), rounded up to whole words.

    Returns:
        The list of chunks; empty when ``text`` contains no tokens.

    Raises:
        ValueError: If ``size``/``overlap`` are outside their valid ranges.
    """
    if size <= 0:
        raise ValueError("size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size.")

    words = text.split()
    if not words:
        return []

    total = len(words)
    chunks: list[str] = []
    start = 0
    while start < total:
        # Greedily pack whole words into the current chunk up to `size` chars.
        # The first word is always taken, even if it alone exceeds `size`, so a
        # single long token is never split.
        end = start
        length = 0
        while end < total:
            addition = len(words[end]) + (1 if end > start else 0)
            if length + addition > size and end > start:
                break
            length += addition
            end += 1

        chunks.append(" ".join(words[start:end]))
        if end >= total:
            break

        if overlap == 0:
            start = end
            continue

        # Step back over whole trailing words until they span at least
        # `overlap` characters, but always advance by at least one word so the
        # loop terminates even for a single oversized token.
        next_start = end - 1
        while next_start > start + 1 and len(" ".join(words[next_start:end])) < overlap:
            next_start -= 1
        start = max(next_start, start + 1)

    return chunks


def chunk_pages(
    pages: list[Page],
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    *,
    words: bool = False,
) -> list[PageChunks]:
    """Chunk every page while preserving its 1-based page number.

    When ``words`` is true, chunking respects whole-word boundaries via
    :func:`chunk_text_words`; otherwise the character-window
    :func:`chunk_text` is used.
    """
    chunker = chunk_text_words if words else chunk_text
    return [
        PageChunks(page_no=page.page_no, chunks=chunker(page.text, size, overlap))
        for page in pages
    ]


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #

_EXIT_USAGE = 2


def _parse_page_range(spec: str) -> tuple[int, int]:
    """Parse a ``--pages`` value like ``"1-3"`` or ``"2"`` into ``(lo, hi)``.

    Raises:
        ValueError: If the spec is malformed or describes an empty range.
    """
    cleaned = spec.strip()
    try:
        if "-" in cleaned:
            lo_str, _, hi_str = cleaned.partition("-")
            lo, hi = int(lo_str), int(hi_str)
        else:
            lo = hi = int(cleaned)
    except ValueError:
        raise ValueError(
            f"invalid --pages value {spec!r}; expected e.g. '1-3' or '2'"
        ) from None

    if lo < 1 or hi < lo:
        raise ValueError(f"invalid --pages value {spec!r}; expected 1 <= start <= end")
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    """Build the ``ingest`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="ingest",
        description=(
            "Extract text from an engineering PDF and split it into overlapping chunks."
        ),
    )
    parser.add_argument("pdf", type=Path, help="Path to the source PDF file.")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a JSON array of {page, chunks} records to stdout.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="N",
        help=f"Maximum characters per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        metavar="N",
        help=f"Characters shared between chunks (default: {DEFAULT_OVERLAP}).",
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help="Chunk on whole-word boundaries so no token is split mid-word.",
    )
    parser.add_argument(
        "--pages",
        metavar="RANGE",
        default=None,
        help="Only process this 1-based page range, e.g. '1-3' or '2'.",
    )
    return parser


def _fail(message: str) -> int:
    """Print an error to stderr and return the usage exit code."""
    print(f"error: {message}", file=sys.stderr)
    return _EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code (0 on success)."""
    args = build_parser().parse_args(argv)

    path: Path = args.pdf
    if not path.exists():
        return _fail(f"file not found: {path}")
    if not path.is_file():
        return _fail(f"not a file: {path}")

    try:
        pdf_bytes = path.read_bytes()
    except OSError as exc:
        return _fail(f"could not read {path}: {exc}")

    try:
        pages = extract_pages(pdf_bytes)
    except ValueError as exc:
        return _fail(str(exc))

    if args.pages is not None:
        try:
            lo, hi = _parse_page_range(args.pages)
        except ValueError as exc:
            return _fail(str(exc))
        pages = [page for page in pages if lo <= page.page_no <= hi]

    try:
        page_chunks = chunk_pages(
            pages, args.chunk_size, args.overlap, words=args.words
        )
    except ValueError as exc:
        return _fail(str(exc))

    if args.as_json:
        payload = [{"page": pc.page_no, "chunks": pc.chunks} for pc in page_chunks]
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for pc in page_chunks:
            print(f"--- page {pc.page_no}: {len(pc.chunks)} chunk(s) ---")
            for index, chunk in enumerate(pc.chunks, start=1):
                preview = " ".join(chunk[:80].split())
                print(f"  [{index}] {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
