"""PDF text extraction and overlapping chunking for TechManualRAG.

The module is intentionally dependency-light: page extraction relies on
PyMuPDF (imported as ``fitz``), while the chunking logic is pure standard
library so it can be unit-tested and reused without pulling in the heavy RAG
stack (LlamaIndex, Streamlit, Ollama, ...).

It also doubles as an offline CLI::

    python ingest.py manual.pdf --json --pages 1-3

Passing ``--index`` additionally embeds the chunks and upserts them into a
local Qdrant collection via :mod:`retrieve`, e.g.::

    python ingest.py manual.pdf --index
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

# Smallest placement, in PDF points, that counts as a figure. A page border, a
# table rule and a one-pixel spacer are all raster images as far as the PDF is
# concerned, and calling them figures would bury the real ones. 24pt is a third
# of an inch: smaller than any diagram worth indexing, larger than any rule.
DEFAULT_MIN_IMAGE_SIZE = 24.0

# Mirrors retrieve.DEFAULT_COLLECTION / retrieve.DEFAULT_QDRANT_PATH, duplicated
# here (rather than imported) so building the CLI parser never requires
# qdrant-client to be installed unless --index is actually used.
_DEFAULT_COLLECTION = "manuals"
_DEFAULT_QDRANT_PATH = "data/qdrant"


@dataclass(frozen=True)
class Page:
    """A single extracted PDF page.

    ``page_no`` is 1-based to match how humans reference document pages.
    """

    page_no: int
    text: str


@dataclass(frozen=True)
class PageImage:
    """One placement of one image on one page.

    A placement, not an image: the same embedded picture can appear on a page
    more than once, and each spot is a separate location a reader might mean.

    ``bbox`` is ``(x0, y0, x1, y1)`` in PDF points, the coordinate system
    PyMuPDF reports and the one a viewer's page coordinates match, so a caller
    can crop or highlight without a second conversion.
    """

    page_no: int
    index: int
    bbox: tuple[float, float, float, float]
    fmt: str
    data: bytes

    @property
    def width(self) -> float:
        """Placement width in PDF points."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Placement height in PDF points."""
        return self.bbox[3] - self.bbox[1]


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


def image_id(page_no: int, index: int) -> str:
    """Stable identifier for one image placement within a document.

    Same shape as :func:`chunk_id` (``"p1-i0"`` beside ``"p1-c0"``) so a later
    indexing step has one convention to key on for both, and so the two can
    never be confused for each other.
    """
    return f"p{page_no}-i{index}"


def extract_images(
    pdf_bytes: bytes,
    min_size: float = DEFAULT_MIN_IMAGE_SIZE,
) -> list[PageImage]:
    """Extract raster image placements from an in-memory PDF.

    Diagrams, schematics and photographed tables are where most of the value in
    a technical manual lives, and text extraction drops all of it. This finds
    where those images sit; embedding them is a separate step.

    Placements are read from the page rather than from the document's image
    table, so an image reused three times on a page yields three results with
    three bounding boxes. Placements are ordered by position, top to bottom then
    left to right, so ``index`` is stable across runs rather than following
    whatever order the PDF happens to store its objects in.

    Two image objects occupying the exact same rectangle are reported once. PDF
    writers routinely emit one picture as several objects, and each object then
    reports every rectangle the picture occupies, so without this the same
    figure comes back squared. Telling a genuine overlay apart from that
    duplication would need pixel comparison, which is out of proportion to how
    often two different images are stacked precisely.

    Args:
        pdf_bytes: Raw bytes of a PDF document.
        min_size: Minimum placement width AND height, in PDF points. Anything
            smaller is a rule, a border or a spacer rather than a figure.

    Returns:
        One :class:`PageImage` per surviving placement, across all pages.

    Raises:
        ValueError: If ``pdf_bytes`` is empty, is not a readable PDF, or
            ``min_size`` is negative.
    """
    if not pdf_bytes:
        raise ValueError("PDF input is empty (0 bytes).")
    if min_size < 0:
        raise ValueError(f"min_size must be >= 0, got {min_size!r}")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several low-level error types.
        raise ValueError(f"Could not open PDF: {exc}") from exc

    out: list[PageImage] = []
    try:
        for page_index, page in enumerate(doc):
            placements = []
            seen: set[tuple[float, float, float, float]] = set()
            for info in page.get_images(full=True):
                xref = info[0]
                for rect in page.get_image_rects(xref):
                    if rect.width < min_size or rect.height < min_size:
                        continue
                    key = (
                        round(rect.x0, 3),
                        round(rect.y0, 3),
                        round(rect.x1, 3),
                        round(rect.y1, 3),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    placements.append((key[1], key[0], rect, xref))

            # Reading order, so index is a property of the page rather than of
            # the file's internal object ordering.
            placements.sort(key=lambda item: (item[0], item[1]))

            for index, (_, _, rect, xref) in enumerate(placements):
                try:
                    extracted = doc.extract_image(xref)
                except Exception:  # A broken or unsupported stream.
                    continue
                out.append(
                    PageImage(
                        page_no=page_index + 1,
                        index=index,
                        bbox=(
                            float(rect.x0),
                            float(rect.y0),
                            float(rect.x1),
                            float(rect.y1),
                        ),
                        fmt=str(extracted.get("ext", "")),
                        data=bytes(extracted.get("image", b"")),
                    )
                )
    finally:
        doc.close()

    return out


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


def chunk_id(page_no: int, index: int) -> str:
    """Return the stable identifier for one chunk within a document.

    The id combines the 1-based ``page_no`` with the 0-based per-page chunk
    ``index`` (e.g. ``"p1-c0"``). It is unique across a document and stable for
    a given extraction/chunking configuration, which is what the roadmap's
    Qdrant indexing step keys each vector on.
    """
    return f"p{page_no}-c{index}"


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
        "--images",
        action="store_true",
        help=(
            "Also report raster image placements (diagrams, schematics, scanned "
            "tables) with their page, id, bounding box in PDF points and format."
        ),
    )
    parser.add_argument(
        "--min-image-size",
        type=float,
        default=DEFAULT_MIN_IMAGE_SIZE,
        metavar="PT",
        help=(
            "Smallest image placement counted as a figure, in PDF points "
            f"(default: {DEFAULT_MIN_IMAGE_SIZE:g}). Below this are page "
            "borders, table rules and spacers."
        ),
    )
    parser.add_argument(
        "--with-ids",
        action="store_true",
        dest="with_ids",
        help=(
            "In --json mode, emit each chunk as a {id, text} object with a "
            "stable id (p{page}-c{index}) instead of a bare string."
        ),
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
    parser.add_argument(
        "--index",
        action="store_true",
        help=(
            "Embed the resulting chunks and upsert them into a local Qdrant "
            "collection (see --collection) for later search with retrieve.py."
        ),
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"Qdrant collection name for --index (default: {_DEFAULT_COLLECTION!r}).",
    )
    parser.add_argument(
        "--qdrant-path",
        default=_DEFAULT_QDRANT_PATH,
        metavar="PATH",
        help=f"On-disk path for the local Qdrant store (default: {_DEFAULT_QDRANT_PATH!r}).",
    )
    return parser


def _fail(message: str) -> int:
    """Print an error to stderr and return the usage exit code."""
    print(f"error: {message}", file=sys.stderr)
    return _EXIT_USAGE


def _image_records(images: list[PageImage]) -> list[dict[str, object]]:
    """Build the ``--json --images`` payload.

    Byte counts, never the bytes: this CLI is meant to stay pipeable, and
    base64 blobs in a JSON stream would make that useless. A caller that wants
    the pixels calls :func:`extract_images` directly.
    """
    return [
        {
            "id": image_id(img.page_no, img.index),
            "page": img.page_no,
            "bbox": list(img.bbox),
            "width": round(img.width, 3),
            "height": round(img.height, 3),
            "format": img.fmt,
            "bytes": len(img.data),
        }
        for img in images
    ]


def _json_records(
    page_chunks: list[PageChunks], *, with_ids: bool
) -> list[dict[str, object]]:
    """Build the ``--json`` payload for ``page_chunks``.

    Without ``with_ids`` each page's ``chunks`` field is a list of raw strings
    (the default, so existing consumers are unaffected). With it, every chunk
    becomes a ``{"id": ..., "text": ...}`` record carrying a stable
    :func:`chunk_id`.
    """
    records: list[dict[str, object]] = []
    for pc in page_chunks:
        chunks: object
        if with_ids:
            chunks = [
                {"id": chunk_id(pc.page_no, index), "text": text}
                for index, text in enumerate(pc.chunks)
            ]
        else:
            chunks = pc.chunks
        records.append({"page": pc.page_no, "chunks": chunks})
    return records


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

    images: list[PageImage] = []
    if args.images:
        try:
            images = extract_images(pdf_bytes, min_size=args.min_image_size)
        except ValueError as exc:
            return _fail(str(exc))
        if args.pages is not None:
            # --pages already narrowed the text; the figures follow it, or the
            # two halves of one report would describe different documents.
            images = [img for img in images if lo <= img.page_no <= hi]

    if args.index:
        # Imported lazily: qdrant-client (and the embedding model it triggers)
        # is only needed when --index is actually requested, keeping plain
        # extraction/chunking usage of this module dependency-light.
        from retrieve import build_index

        build_index(page_chunks, args.collection, path=args.qdrant_path)
        total_chunks = sum(len(pc.chunks) for pc in page_chunks)
        # Printed to stderr so stdout stays clean/parseable when --json is
        # combined with --index.
        print(
            f"Indexed {total_chunks} chunk(s) from {len(page_chunks)} page(s) "
            f"into Qdrant collection {args.collection!r} at {args.qdrant_path!r}.",
            file=sys.stderr,
        )

    if args.as_json:
        payload: object = _json_records(page_chunks, with_ids=args.with_ids)
        if args.images:
            # A second top-level key rather than images nested per page: a
            # consumer that only wants figures should not have to walk the text.
            payload = {"pages": payload, "images": _image_records(images)}
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for pc in page_chunks:
            print(f"--- page {pc.page_no}: {len(pc.chunks)} chunk(s) ---")
            for index, chunk in enumerate(pc.chunks, start=1):
                preview = " ".join(chunk[:80].split())
                print(f"  [{index}] {preview}")
        if args.images:
            print(f"--- {len(images)} image(s) ---")
            for img in images:
                x0, y0, x1, y1 = img.bbox
                print(
                    f"  [{image_id(img.page_no, img.index)}] page {img.page_no} "
                    f"{img.fmt or '?'} {img.width:.0f}x{img.height:.0f}pt "
                    f"at ({x0:.0f}, {y0:.0f})-({x1:.0f}, {y1:.0f}) "
                    f"{len(img.data)} bytes"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
