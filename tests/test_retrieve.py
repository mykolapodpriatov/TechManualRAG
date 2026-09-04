"""Tests for the local Qdrant indexing/search helpers in ``retrieve.py``.

These tests never touch the network or download a real embedding model: they
inject a tiny deterministic stub embedder and use Qdrant's embedded, on-disk
mode (a ``tmp_path`` directory) instead of a running Qdrant server. This
mirrors ``test_ingest.py`` staying independent of the heavy RAG stack, and
keeps the suite runnable with just ``pytest``, ``PyMuPDF``, and
``qdrant-client`` installed.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import pytest

from ingest import Page, PageChunks, chunk_id, chunk_pages
from retrieve import SearchResult, build_index, embed_chunks, index_pages, search

_STUB_DIMS = 32


def _stub_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words embedder for tests.

    Hashes each lowercased whitespace token into one of ``_STUB_DIMS``
    buckets and counts occurrences, then L2-normalises. Texts that share
    vocabulary score higher under cosine similarity than unrelated ones,
    which is enough to exercise ranking without a real model.
    """
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _STUB_DIMS
        for token in text.lower().split():
            vec[zlib.crc32(token.encode("utf-8")) % _STUB_DIMS] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


@pytest.fixture
def qdrant_path(tmp_path: Path) -> str:
    return str(tmp_path / "qdrant-store")


TORQUE_CHUNKS = PageChunks(
    page_no=1,
    chunks=["Torque spec for the head bolts: tighten to 35 Nm in a star pattern."],
)
UNRELATED_CHUNKS = PageChunks(
    page_no=2,
    chunks=["Warranty terms and customer support contact information."],
)


# --------------------------------------------------------------------------- #
# embed_chunks
# --------------------------------------------------------------------------- #


def test_embed_chunks_empty_input_returns_empty_list_without_calling_embedder() -> None:
    def boom(_: list[str]) -> list[list[float]]:
        raise AssertionError("embedder should not be called for empty input")

    assert embed_chunks([], embedder=boom) == []


def test_embed_chunks_uses_injected_embedder() -> None:
    vectors = embed_chunks(["a", "b", "c"], embedder=_stub_embedder)
    assert len(vectors) == 3
    assert all(len(v) == _STUB_DIMS for v in vectors)


# --------------------------------------------------------------------------- #
# build_index / search round trip
# --------------------------------------------------------------------------- #


def test_search_ranks_relevant_chunk_first(qdrant_path: str) -> None:
    build_index(
        [TORQUE_CHUNKS, UNRELATED_CHUNKS],
        "manuals",
        path=qdrant_path,
        embedder=_stub_embedder,
    )

    results = search(
        "torque spec", "manuals", path=qdrant_path, embedder=_stub_embedder
    )

    assert results
    assert isinstance(results[0], SearchResult)
    assert "Torque spec" in results[0].text
    assert results[0].page_no == 1


def test_search_returns_page_numbers_and_scores(qdrant_path: str) -> None:
    build_index(
        [TORQUE_CHUNKS, UNRELATED_CHUNKS],
        "manuals",
        path=qdrant_path,
        embedder=_stub_embedder,
    )

    results = search(
        "torque spec", "manuals", path=qdrant_path, embedder=_stub_embedder
    )

    assert {r.page_no for r in results} <= {1, 2}
    assert all(isinstance(r.score, float) for r in results)
    # Scores are sorted best-first.
    assert list(r.score for r in results) == sorted(
        (r.score for r in results), reverse=True
    )


def test_search_respects_top_k(qdrant_path: str) -> None:
    many_chunks = PageChunks(
        page_no=3,
        chunks=[
            f"Procedure step {i}: check the pressure relief valve." for i in range(10)
        ],
    )
    build_index([many_chunks], "manuals", path=qdrant_path, embedder=_stub_embedder)

    results = search(
        "pressure relief valve",
        "manuals",
        path=qdrant_path,
        top_k=3,
        embedder=_stub_embedder,
    )

    assert len(results) == 3


def test_search_on_missing_collection_returns_empty_list(qdrant_path: str) -> None:
    assert (
        search("anything", "never-indexed", path=qdrant_path, embedder=_stub_embedder)
        == []
    )


def test_build_index_empty_page_chunks_is_a_noop(qdrant_path: str) -> None:
    build_index([], "manuals", path=qdrant_path, embedder=_stub_embedder)
    assert (
        search("anything", "manuals", path=qdrant_path, embedder=_stub_embedder) == []
    )


def test_build_index_is_idempotent_for_unchanged_chunks(qdrant_path: str) -> None:
    build_index([TORQUE_CHUNKS], "manuals", path=qdrant_path, embedder=_stub_embedder)
    build_index([TORQUE_CHUNKS], "manuals", path=qdrant_path, embedder=_stub_embedder)

    results = search(
        "torque", "manuals", path=qdrant_path, top_k=10, embedder=_stub_embedder
    )

    # Re-indexing the same chunk ids upserts in place rather than duplicating.
    assert len(results) == 1


def test_build_index_updates_existing_chunk_text(qdrant_path: str) -> None:
    original = PageChunks(page_no=1, chunks=["Old content about warranty terms."])
    updated = PageChunks(page_no=1, chunks=["New content about torque specifications."])

    build_index([original], "manuals", path=qdrant_path, embedder=_stub_embedder)
    build_index([updated], "manuals", path=qdrant_path, embedder=_stub_embedder)

    results = search(
        "torque", "manuals", path=qdrant_path, top_k=10, embedder=_stub_embedder
    )

    assert len(results) == 1
    assert results[0].text == "New content about torque specifications."


def test_search_pages_restricts_hits_to_range(qdrant_path: str) -> None:
    three_pages = [
        PageChunks(page_no=1, chunks=["Torque spec page one about head bolts."]),
        PageChunks(page_no=2, chunks=["Torque spec page two about lock nuts."]),
        PageChunks(page_no=3, chunks=["Torque spec page three about washers."]),
    ]
    build_index(three_pages, "manuals", path=qdrant_path, embedder=_stub_embedder)

    only_two = search(
        "torque spec",
        "manuals",
        path=qdrant_path,
        top_k=10,
        pages=(2, 2),
        embedder=_stub_embedder,
    )
    assert only_two
    assert {result.page_no for result in only_two} == {2}

    unfiltered = search(
        "torque spec",
        "manuals",
        path=qdrant_path,
        top_k=10,
        embedder=_stub_embedder,
    )
    assert {result.page_no for result in unfiltered} == {1, 2, 3}
    assert len(unfiltered) == 3


def test_search_min_score_drops_weak_hits(qdrant_path: str) -> None:
    # Unit vectors so cosine equals the first component against query [1, 0]:
    # strong -> 0.9, weak -> 0.1.
    query_text = "query"
    strong_text = "strong"
    weak_text = "weak"
    table = {
        query_text: [1.0, 0.0],
        strong_text: [0.9, 0.19**0.5],
        weak_text: [0.1, 0.99**0.5],
    }

    def known_embedder(texts: list[str]) -> list[list[float]]:
        return [table[text] for text in texts]

    build_index(
        [
            PageChunks(page_no=1, chunks=[strong_text]),
            PageChunks(page_no=2, chunks=[weak_text]),
        ],
        "manuals",
        path=qdrant_path,
        embedder=known_embedder,
    )

    kept = search(
        query_text,
        "manuals",
        path=qdrant_path,
        top_k=5,
        min_score=0.5,
        embedder=known_embedder,
    )
    assert len(kept) == 1
    assert kept[0].text == strong_text
    assert kept[0].score == pytest.approx(0.9, abs=1e-5)

    both = search(
        query_text,
        "manuals",
        path=qdrant_path,
        top_k=5,
        min_score=None,
        embedder=known_embedder,
    )
    assert {result.text for result in both} == {strong_text, weak_text}
    by_text = {result.text: result.score for result in both}
    assert by_text[strong_text] == pytest.approx(0.9, abs=1e-5)
    assert by_text[weak_text] == pytest.approx(0.1, abs=1e-5)


def test_search_min_score_rejects_out_of_range(qdrant_path: str) -> None:
    with pytest.raises(ValueError, match="min_score"):
        search(
            "torque",
            "manuals",
            path=qdrant_path,
            min_score=-0.1,
            embedder=_stub_embedder,
        )
    with pytest.raises(ValueError, match="min_score"):
        search(
            "torque",
            "manuals",
            path=qdrant_path,
            min_score=1.1,
            embedder=_stub_embedder,
        )


def test_search_pages_rejects_inverted_range(qdrant_path: str) -> None:
    build_index([TORQUE_CHUNKS], "manuals", path=qdrant_path, embedder=_stub_embedder)
    with pytest.raises(ValueError, match="pages"):
        search(
            "torque spec",
            "manuals",
            path=qdrant_path,
            pages=(3, 1),
            embedder=_stub_embedder,
        )


def test_build_index_persists_across_client_instances(qdrant_path: str) -> None:
    # build_index and search each open/close their own client; this checks
    # the on-disk store genuinely persists between those separate sessions.
    build_index([TORQUE_CHUNKS], "manuals", path=qdrant_path, embedder=_stub_embedder)
    results = search("torque", "manuals", path=qdrant_path, embedder=_stub_embedder)
    assert len(results) == 1


# --------------------------------------------------------------------------- #
# index_pages
# --------------------------------------------------------------------------- #


def _page(page_no: int, text: str) -> Page:
    return Page(page_no=page_no, text=text)


def test_index_pages_makes_pages_searchable(qdrant_path: str) -> None:
    """The whole path the UI uses: extracted pages in, searchable chunks out."""
    pages = [
        _page(1, "Torque spec for the head bolts: tighten to 35 Nm in a star pattern."),
        _page(2, "Warranty terms and customer support contact information."),
    ]

    indexed = index_pages(
        pages,
        "manuals",
        qdrant_path,
        chunk_size=200,
        overlap=0,
        embedder=_stub_embedder,
    )
    assert indexed == 2

    hits = search(
        "torque spec head bolts", "manuals", path=qdrant_path, embedder=_stub_embedder
    )
    assert hits
    assert hits[0].page_no == 1


def test_index_pages_returns_the_chunk_count(qdrant_path: str) -> None:
    """The count is what the UI reports, so it must match what was stored."""
    long_page = _page(1, "abcdefghij " * 20)

    indexed = index_pages(
        [long_page],
        "manuals",
        qdrant_path,
        chunk_size=50,
        overlap=10,
        embedder=_stub_embedder,
    )

    expected = sum(len(pc.chunks) for pc in chunk_pages([long_page], 50, 10))
    assert indexed == expected > 1


def test_index_pages_agrees_with_the_cli_chunk_ids(qdrant_path: str) -> None:
    """UI and CLI must key the same document on the same ids, or a re-index
    through the other path would duplicate every point instead of updating it."""
    pages = [_page(1, "Hydraulic pump X100 overview and safety notices." * 5)]

    index_pages(
        pages,
        "manuals",
        qdrant_path,
        chunk_size=60,
        overlap=10,
        embedder=_stub_embedder,
    )
    hits = search(
        "hydraulic pump", "manuals", top_k=50, path=qdrant_path, embedder=_stub_embedder
    )

    expected_ids = {
        chunk_id(pc.page_no, index)
        for pc in chunk_pages(pages, 60, 10)
        for index in range(len(pc.chunks))
    }
    assert {hit.chunk_id for hit in hits} == expected_ids


def test_index_pages_is_idempotent(qdrant_path: str) -> None:
    """Indexing the same document twice updates points instead of duplicating."""
    pages = [_page(1, "Torque spec for the head bolts: tighten to 35 Nm.")]

    first = index_pages(
        pages,
        "manuals",
        qdrant_path,
        chunk_size=200,
        overlap=0,
        embedder=_stub_embedder,
    )
    second = index_pages(
        pages,
        "manuals",
        qdrant_path,
        chunk_size=200,
        overlap=0,
        embedder=_stub_embedder,
    )
    assert first == second == 1

    hits = search(
        "torque", "manuals", top_k=50, path=qdrant_path, embedder=_stub_embedder
    )
    assert len(hits) == 1


def test_index_pages_on_a_textless_document_writes_nothing(qdrant_path: str) -> None:
    """A scanned PDF has no extractable text; that is a warning, not a crash,
    and it must not leave an empty collection behind."""
    indexed = index_pages(
        [_page(1, ""), _page(2, "")],
        "manuals",
        qdrant_path,
        embedder=_stub_embedder,
    )

    assert indexed == 0
    assert (
        search("anything", "manuals", path=qdrant_path, embedder=_stub_embedder) == []
    )


def test_index_pages_rejects_bad_chunking(qdrant_path: str) -> None:
    """Invalid chunk settings surface as ValueError for the UI to display."""
    with pytest.raises(ValueError):
        index_pages(
            [_page(1, "some text")],
            "manuals",
            qdrant_path,
            chunk_size=10,
            overlap=10,
            embedder=_stub_embedder,
        )
