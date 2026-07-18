"""Tests for the offline ``ingest`` CLI.

``main(argv)`` is invoked directly (no subprocess) so coverage tools and the
debugger work normally. The fixture PDF is generated in-memory with PyMuPDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from ingest import chunk_id, main

PAGE_TEXTS = [
    "Section 1 Overview of the hydraulic assembly and safety notes.",
    "Section 2 Torque table and step-by-step tightening procedure.",
    "Section 3 Troubleshooting the pressure relief valve.",
]


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    doc = fitz.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            page.insert_text((72, 72), text, fontsize=11)
        doc.save(str(path))
    finally:
        doc.close()


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "manual.pdf"
    _write_pdf(path, PAGE_TEXTS)
    return path


def test_json_output_is_valid(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(pdf_path), "--json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert [record["page"] for record in payload] == [1, 2, 3]
    assert all(isinstance(record["chunks"], list) for record in payload)
    assert "hydraulic assembly" in " ".join(payload[0]["chunks"])


def test_pages_range_filter(pdf_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(pdf_path), "--json", "--pages", "2-3"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert [record["page"] for record in payload] == [2, 3]


def test_single_page_filter(pdf_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(pdf_path), "--json", "--pages", "1"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert [record["page"] for record in payload] == [1]


def test_chunk_size_and_overlap_flags(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            str(pdf_path),
            "--json",
            "--pages",
            "1",
            "--chunk-size",
            "20",
            "--overlap",
            "5",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    chunks = payload[0]["chunks"]
    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_human_readable_output(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(pdf_path)])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "page 1" in out
    assert "chunk(s)" in out


def test_missing_file_exits_nonzero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["/no/such/file.pdf", "--json"])
    assert exit_code != 0
    assert "not found" in capsys.readouterr().err


def test_non_pdf_input_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bogus = tmp_path / "notes.txt"
    bogus.write_text("just some plain text, definitely not a PDF")

    exit_code = main([str(bogus), "--json"])
    assert exit_code != 0
    assert capsys.readouterr().err.strip() != ""


def test_directory_input_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(tmp_path), "--json"])
    assert exit_code != 0
    assert "not a file" in capsys.readouterr().err


def test_bad_pages_value_exits_nonzero(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(pdf_path), "--pages", "3-1"])
    assert exit_code != 0
    assert "--pages" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Stable chunk ids (--with-ids)
# --------------------------------------------------------------------------- #


def test_chunk_id_format_is_stable() -> None:
    assert chunk_id(1, 0) == "p1-c0"
    assert chunk_id(3, 12) == "p3-c12"
    # Same inputs always yield the same id.
    assert chunk_id(2, 5) == chunk_id(2, 5)


def test_chunk_id_is_unique_across_page_and_index() -> None:
    ids = [chunk_id(page, index) for page in range(1, 6) for index in range(4)]
    assert len(ids) == len(set(ids))


def test_default_json_chunks_are_bare_strings(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(pdf_path), "--json"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    for record in payload:
        assert all(isinstance(chunk, str) for chunk in record["chunks"])


def test_with_ids_emits_id_and_text_objects(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [str(pdf_path), "--json", "--with-ids", "--chunk-size", "20", "--overlap", "5"]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    all_ids: list[str] = []
    for record in payload:
        page_no = record["page"]
        for index, chunk in enumerate(record["chunks"]):
            assert set(chunk) == {"id", "text"}
            assert isinstance(chunk["text"], str)
            assert chunk["id"] == f"p{page_no}-c{index}"
            all_ids.append(chunk["id"])

    # Ids are unique across the whole document.
    assert len(all_ids) == len(set(all_ids))
    assert all_ids  # the fixture yields at least one chunk


def test_with_ids_preserves_chunk_text(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(pdf_path), "--json", "--with-ids", "--pages", "1"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    joined = " ".join(chunk["text"] for chunk in payload[0]["chunks"])
    assert "hydraulic assembly" in joined
