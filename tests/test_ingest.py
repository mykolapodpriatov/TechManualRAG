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
    chunk_id,
    chunk_pages,
    chunk_text,
    chunk_text_words,
    extract_images,
    extract_pages,
    image_id,
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
        assert (
            "".join(c if i == 0 else c[5:] for i, c in enumerate(pc.chunks))
            == page.text
        )


# --------------------------------------------------------------------------- #
# Word-boundary chunking (chunk_text_words)
# --------------------------------------------------------------------------- #


def _common_word_overlap(first: str, second: str) -> list[str]:
    """Return the longest whole-word suffix of ``first`` that prefixes ``second``."""
    first_words = first.split()
    second_words = second.split()
    best: list[str] = []
    limit = min(len(first_words), len(second_words))
    for k in range(1, limit + 1):
        if first_words[-k:] == second_words[:k]:
            best = second_words[:k]
    return best


def test_chunk_text_words_empty_returns_no_chunks() -> None:
    assert chunk_text_words("") == []
    assert chunk_text_words("   \n\t  ") == []


def test_chunk_text_words_short_text_single_chunk() -> None:
    assert chunk_text_words("short manual", size=512, overlap=50) == ["short manual"]


def test_chunk_text_words_normalises_internal_whitespace() -> None:
    chunks = chunk_text_words("alpha\n\tbeta   gamma", size=512, overlap=0)
    assert chunks == ["alpha beta gamma"]


def test_chunk_text_words_never_splits_a_word() -> None:
    # Distinct, intact tokens; a mid-word cut would produce a token absent
    # from this set, which the assertion below would catch.
    words = [f"word{i:03d}" for i in range(200)]
    text = " ".join(words)
    original = set(words)

    chunks = chunk_text_words(text, size=50, overlap=14)

    assert len(chunks) > 1
    for chunk in chunks:
        tokens = chunk.split()
        assert tokens, "no chunk should be empty"
        # No leading/trailing whitespace and no collapsed double spaces.
        assert chunk == " ".join(tokens)
        # Every token is an intact original word (never a fragment).
        assert all(token in original for token in tokens)


def test_chunk_text_words_stays_within_size_when_words_allow() -> None:
    words = [f"tok{i:02d}" for i in range(80)]  # each token is 5 chars
    text = " ".join(words)
    size = 40

    chunks = chunk_text_words(text, size=size, overlap=10)

    # No individual token exceeds `size`, so every chunk must fit.
    assert all(len(chunk) <= size for chunk in chunks)


def test_chunk_text_words_overlap_shares_whole_words() -> None:
    words = [f"token{i:04d}" for i in range(120)]  # each token is 9 chars
    text = " ".join(words)
    overlap = 20

    chunks = chunk_text_words(text, size=60, overlap=overlap)

    assert len(chunks) > 1
    for first, second in zip(chunks[:-1], chunks[1:], strict=True):
        shared = _common_word_overlap(first, second)
        assert shared, "consecutive chunks must share at least one whole word"
        # The shared whole words span at least the requested overlap.
        assert len(" ".join(shared)) >= overlap


def test_chunk_text_words_zero_overlap_has_no_shared_words() -> None:
    words = [f"item{i:03d}" for i in range(60)]
    text = " ".join(words)

    chunks = chunk_text_words(text, size=40, overlap=0)

    assert len(chunks) > 1
    for first, second in zip(chunks[:-1], chunks[1:], strict=True):
        assert _common_word_overlap(first, second) == []


def test_chunk_text_words_oversized_token_becomes_its_own_chunk() -> None:
    long_token = "X" * 100
    text = f"alpha {long_token} beta"

    chunks = chunk_text_words(text, size=20, overlap=5)

    # The long token is emitted whole rather than being cut.
    assert long_token in chunks
    for chunk in chunks:
        for token in chunk.split():
            assert token in {"alpha", long_token, "beta"}


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 150)],
)
def test_chunk_text_words_rejects_bad_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text_words("some technical text", size=size, overlap=overlap)


def test_chunk_text_words_defaults_match_module_constants() -> None:
    text = " ".join(f"w{i:04d}" for i in range(400))
    chunks = chunk_text_words(text)
    assert all(len(chunk) <= DEFAULT_CHUNK_SIZE for chunk in chunks)
    for first, second in zip(chunks[:-1], chunks[1:], strict=True):
        shared = _common_word_overlap(first, second)
        assert len(" ".join(shared)) >= DEFAULT_OVERLAP


def test_chunk_pages_words_mode_routes_through_word_chunker(
    two_page_pdf: bytes,
) -> None:
    pages = extract_pages(two_page_pdf)

    char_chunks = chunk_pages(pages, size=20, overlap=5)
    word_chunks = chunk_pages(pages, size=20, overlap=5, words=True)

    # Word mode: every token in every chunk is an intact source word.
    for page, pc in zip(pages, word_chunks, strict=True):
        original = set(page.text.split())
        for chunk in pc.chunks:
            assert all(token in original for token in chunk.split())

    # Char mode over the same small pages splits at least one word, proving the
    # two strategies genuinely differ.
    char_fragments = any(
        token not in set(page.text.split())
        for page, pc in zip(pages, char_chunks, strict=True)
        for chunk in pc.chunks
        for token in chunk.split()
    )
    assert char_fragments


# --------------------------------------------------------------------------- #
# extract_images
# --------------------------------------------------------------------------- #


def _png_bytes(
    width: int, height: int, colour: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    """A solid-colour PNG, built with PyMuPDF so no binary is checked in."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, colour)
    return pix.tobytes("png")


def _pdf_with_images(
    placements: list[list[tuple[float, float, float, float]]],
    *,
    image_size: int = 64,
) -> bytes:
    """Render a PDF where page ``i`` carries ``placements[i]`` image rectangles."""
    png = _png_bytes(image_size, image_size)
    doc = fitz.open()
    try:
        for rects in placements:
            page = doc.new_page()
            page.insert_text((72, 720), "caption", fontsize=11)
            for rect in rects:
                page.insert_image(fitz.Rect(*rect), stream=png)
        return doc.tobytes()
    finally:
        doc.close()


def test_finds_a_placement_with_its_page_and_box() -> None:
    pdf = _pdf_with_images([[(100.0, 200.0, 300.0, 400.0)]])

    images = extract_images(pdf)

    assert len(images) == 1
    img = images[0]
    assert img.page_no == 1
    assert img.index == 0
    assert img.bbox == pytest.approx((100.0, 200.0, 300.0, 400.0), abs=0.5)
    assert img.width == pytest.approx(200.0, abs=0.5)
    assert img.height == pytest.approx(200.0, abs=0.5)
    assert img.fmt == "png"
    assert len(img.data) > 0


def test_the_same_image_placed_twice_yields_two_placements() -> None:
    """A placement, not an image: each spot is a location a reader might mean."""
    pdf = _pdf_with_images([[(50.0, 50.0, 150.0, 150.0), (50.0, 300.0, 150.0, 400.0)]])

    images = extract_images(pdf)

    assert len(images) == 2
    assert [img.index for img in images] == [0, 1]
    assert images[0].bbox[1] < images[1].bbox[1]


def test_placements_are_ordered_top_to_bottom_then_left_to_right() -> None:
    """Index has to be a property of the page, not of the PDF's object order."""
    pdf = _pdf_with_images(
        [
            [
                (300.0, 400.0, 400.0, 500.0),  # bottom right
                (100.0, 100.0, 200.0, 200.0),  # top left
                (300.0, 100.0, 400.0, 200.0),  # top right
            ]
        ]
    )

    images = extract_images(pdf)

    assert [(round(i.bbox[1]), round(i.bbox[0])) for i in images] == [
        (100, 100),
        (100, 300),
        (400, 300),
    ]


def test_ids_are_stable_across_runs() -> None:
    pdf = _pdf_with_images(
        [[(100.0, 100.0, 200.0, 200.0)], [(50.0, 50.0, 200.0, 200.0)]]
    )

    first = [image_id(i.page_no, i.index) for i in extract_images(pdf)]
    second = [image_id(i.page_no, i.index) for i in extract_images(pdf)]

    assert first == second == ["p1-i0", "p2-i0"]


def test_image_id_matches_the_chunk_id_shape() -> None:
    """One convention for both, and never confusable for each other."""
    assert image_id(3, 1) == "p3-i1"
    assert chunk_id(3, 1) == "p3-c1"


def test_a_page_with_no_images_contributes_nothing() -> None:
    pdf = _pdf_with_images([[], [(100.0, 100.0, 200.0, 200.0)]])

    images = extract_images(pdf)

    assert [img.page_no for img in images] == [2]


def test_specks_below_the_size_floor_are_not_figures() -> None:
    """A page border and a table rule are raster images too."""
    pdf = _pdf_with_images(
        [
            [
                (100.0, 100.0, 105.0, 400.0),  # a 5pt-wide rule
                (100.0, 500.0, 300.0, 700.0),  # a real figure
            ]
        ]
    )

    images = extract_images(pdf)

    assert len(images) == 1
    assert images[0].width == pytest.approx(200.0, abs=0.5)


def test_the_size_floor_is_configurable() -> None:
    pdf = _pdf_with_images([[(100.0, 100.0, 130.0, 130.0)]])

    assert extract_images(pdf, min_size=40.0) == []
    assert len(extract_images(pdf, min_size=10.0)) == 1


def test_empty_input_and_a_negative_floor_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_images(b"")
    with pytest.raises(ValueError, match="min_size"):
        extract_images(_pdf_with_images([[]]), min_size=-1.0)


def test_a_non_pdf_is_rejected() -> None:
    with pytest.raises(ValueError, match="Could not open PDF"):
        extract_images(b"not a pdf at all")


def test_a_text_only_pdf_has_no_images() -> None:
    assert extract_images(_make_pdf([PAGE_ONE_TEXT, PAGE_TWO_TEXT])) == []
