"""Tests for the PDF ingestion helpers.

The fixture PDF is generated in-memory with PyMuPDF at test time, so no binary
blob is checked into the repository. ``app.py`` is intentionally never imported
here (it would pull in Streamlit); ``ingest.py`` is exercised directly.
"""

from __future__ import annotations

import fitz
import pytest

from ingest import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Page,
    chunk_pages,
    chunk_text,
    extract_pages,
)

PAGE_ONE_TEXT = "Hydraulic Pump Model X100\nMaintenance Schedule"
PAGE_TWO_TEXT = "Torque specifications\nStep 5 tightening sequence"


def _make_pdf(page_texts: list[str]) -> bytes:
    """Render a small PDF in memory, one page per entry in ``page_texts``."""
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text, fontsize=11)
        return doc.tobytes()
    finally:
        doc.close()


@pytest.fixture
def two_page_pdf() -> bytes:
    return _make_pdf([PAGE_ONE_TEXT, PAGE_TWO_TEXT])


def test_extract_pages_counts_pages(two_page_pdf: bytes) -> None:
    pages = extract_pages(two_page_pdf)
    assert [p.page_no for p in pages] == [1, 2]


def test_extract_pages_returns_text(two_page_pdf: bytes) -> None:
    pages = extract_pages(two_page_pdf)
    assert "Hydraulic Pump Model X100" in pages[0].text
    assert "Maintenance Schedule" in pages[0].text
    assert "Torque specifications" in pages[1].text
    # Trailing whitespace from PyMuPDF must be stripped.
    assert pages[0].text == pages[0].text.strip()


def test_extract_pages_empty_page_yields_empty_text() -> None:
    pdf = _make_pdf([""])  # a single page with no drawn text
    pages = extract_pages(pdf)
    assert len(pages) == 1
    assert pages[0] == Page(page_no=1, text="")


def test_extract_pages_rejects_zero_byte_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_pages(b"")


def test_extract_pages_rejects_non_pdf_input() -> None:
    with pytest.raises(ValueError, match="Could not open PDF"):
        extract_pages(b"this is definitely not a pdf")


def test_chunk_text_empty_returns_no_chunks() -> None:
    assert chunk_text("") == []


def test_chunk_text_short_text_single_chunk() -> None:
    chunks = chunk_text("short", size=512, overlap=50)
    assert chunks == ["short"]


def test_chunk_text_chunk_count_is_correct() -> None:
    text = "A" * 1000
    size, overlap = 512, 50
    chunks = chunk_text(text, size=size, overlap=overlap)
    # step = 462 -> starts at 0, 462, 924 => 3 chunks.
    assert len(chunks) == 3
    assert chunks[0] == text[0:512]
    assert chunks[1] == text[462:974]
    assert chunks[2] == text[924:1000]


def test_chunk_text_consecutive_chunks_overlap() -> None:
    # Use distinct characters so overlap comparison is meaningful.
    text = "".join(chr(0x41 + (i % 26)) for i in range(2000))
    size, overlap = 300, 40
    chunks = chunk_text(text, size=size, overlap=overlap)
    assert len(chunks) > 1
    for first, second in zip(chunks[:-1], chunks[1:], strict=True):
        # Every non-final chunk is full length, so the overlap is exact.
        assert first[-overlap:] == second[:overlap]


def test_chunk_text_default_overlap_is_50() -> None:
    text = "B" * 2000
    chunks = chunk_text(text)
    for first, second in zip(chunks[:-1], chunks[1:], strict=True):
        assert first[-DEFAULT_OVERLAP:] == second[:DEFAULT_OVERLAP]
    assert all(len(c) <= DEFAULT_CHUNK_SIZE for c in chunks)


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 150)],
)
def test_chunk_text_rejects_bad_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)


def test_chunk_pages_preserves_page_metadata(two_page_pdf: bytes) -> None:
    pages = extract_pages(two_page_pdf)
    page_chunks = chunk_pages(pages, size=20, overlap=5)
    assert [pc.page_no for pc in page_chunks] == [1, 2]
    # Each page's chunks re-join (accounting for overlap) to the source text.
    for page, pc in zip(pages, page_chunks, strict=True):
        assert pc.chunks[0] == page.text[:20]
        assert "".join(
            c if i == 0 else c[5:] for i, c in enumerate(pc.chunks)
        ) == page.text
