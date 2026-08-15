"""Tests for the offline ``retrieve`` CLI.

``main(argv, embedder=...)`` is invoked directly (no subprocess) so coverage
tools and the debugger work normally. The embedding backend is the same
deterministic stub used by ``test_retrieve.py`` — the real BGE model is
never loaded.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest

from ingest import PageChunks
from retrieve import build_index, main

_STUB_DIMS = 32

TORQUE_CHUNKS = PageChunks(
    page_no=1,
    chunks=["Torque spec for the head bolts: tighten to 35 Nm in a star pattern."],
)
UNRELATED_CHUNKS = PageChunks(
    page_no=2,
    chunks=["Warranty terms and customer support contact information."],
)


def _stub_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words embedder — same scheme as ``test_retrieve``."""
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


@pytest.fixture
def indexed_store(qdrant_path: str) -> str:
    build_index(
        [TORQUE_CHUNKS, UNRELATED_CHUNKS],
        "manuals",
        path=qdrant_path,
        embedder=_stub_embedder,
    )
    return qdrant_path


def test_json_output_is_valid(
    indexed_store: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["torque spec", "--qdrant-path", indexed_store, "--json"],
        embedder=_stub_embedder,
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload
    hit = payload[0]
    assert set(hit) == {"id", "page", "text", "score"}
    assert hit["id"] == "p1-c0"
    assert hit["page"] == 1
    assert "Torque spec" in hit["text"]
    assert isinstance(hit["score"], float)


def test_human_readable_output(
    indexed_store: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["torque spec", "--qdrant-path", indexed_store],
        embedder=_stub_embedder,
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "page 1" in out
    assert "score" in out
    assert "Torque spec" in out


def test_top_k_flag(indexed_store: str, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["torque spec", "--qdrant-path", indexed_store, "--top-k", "1", "--json"],
        embedder=_stub_embedder,
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1


def test_collection_flag_targets_named_collection(
    qdrant_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(
        [TORQUE_CHUNKS],
        "my-manuals",
        path=qdrant_path,
        embedder=_stub_embedder,
    )

    exit_code = main(
        [
            "torque spec",
            "--collection",
            "my-manuals",
            "--qdrant-path",
            qdrant_path,
            "--json",
        ],
        embedder=_stub_embedder,
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["page"] == 1


def test_missing_collection_exits_nonzero(
    qdrant_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["torque spec", "--qdrant-path", qdrant_path],
        embedder=_stub_embedder,
    )
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "does not exist" in err
    assert "ingest.py" in err
    assert "--index" in err


def test_empty_query_exits_nonzero(
    indexed_store: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["", "--qdrant-path", indexed_store],
        embedder=_stub_embedder,
    )
    assert exit_code != 0
    assert "empty" in capsys.readouterr().err


def test_whitespace_query_exits_nonzero(
    indexed_store: str, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["   ", "--qdrant-path", indexed_store],
        embedder=_stub_embedder,
    )
    assert exit_code != 0
    assert "empty" in capsys.readouterr().err


def test_json_is_an_array_even_when_no_chunks_match(
    qdrant_path: str, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_page = PageChunks(page_no=1, chunks=["zzzz unused vocabulary"])
    build_index([empty_page], "manuals", path=qdrant_path, embedder=_stub_embedder)

    exit_code = main(
        ["qqqq", "--qdrant-path", qdrant_path, "--json", "--top-k", "1"],
        embedder=_stub_embedder,
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
