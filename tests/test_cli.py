"""Tests for the offline ``ingest`` CLI.

``main(argv)`` is invoked directly (no subprocess) so coverage tools and the
debugger work normally. The fixture PDF is generated in-memory with PyMuPDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest

from ingest import main

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


def test_json_output_is_valid(pdf_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
        [str(pdf_path), "--json", "--pages", "1", "--chunk-size", "20", "--overlap", "5"]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    chunks = payload[0]["chunks"]
    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_human_readable_output(pdf_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
