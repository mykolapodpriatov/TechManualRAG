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


# --------------------------------------------------------------------------- #
# --index wiring (retrieve.build_index itself is exercised in test_retrieve.py;
# these tests only check that the CLI calls it with the right arguments,
# stubbed out so they never need sentence-transformers/qdrant network access).
# --------------------------------------------------------------------------- #


def test_index_flag_calls_build_index_and_reports_to_stderr(
    pdf_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import retrieve

    calls: list[tuple[list, str, str]] = []

    def fake_build_index(
        page_chunks, collection_name, path=retrieve.DEFAULT_QDRANT_PATH, **_
    ):
        calls.append((page_chunks, collection_name, path))

    monkeypatch.setattr(retrieve, "build_index", fake_build_index)

    qdrant_path = str(tmp_path / "qdrant-store")
    exit_code = main(
        [
            str(pdf_path),
            "--index",
            "--collection",
            "my-manuals",
            "--qdrant-path",
            qdrant_path,
        ]
    )
    assert exit_code == 0

    assert len(calls) == 1
    page_chunks, collection_name, path = calls[0]
    assert collection_name == "my-manuals"
    assert path == qdrant_path
    assert [pc.page_no for pc in page_chunks] == [1, 2, 3]

    err = capsys.readouterr().err
    assert "Indexed" in err
    assert "my-manuals" in err


def test_index_flag_does_not_pollute_json_stdout(
    pdf_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import retrieve

    monkeypatch.setattr(retrieve, "build_index", lambda *a, **k: None)

    qdrant_path = str(tmp_path / "qdrant-store")
    exit_code = main([str(pdf_path), "--index", "--json", "--qdrant-path", qdrant_path])
    assert exit_code == 0

    out = capsys.readouterr().out
    payload = json.loads(
        out
    )  # would raise if the index confirmation leaked into stdout
    assert [record["page"] for record in payload] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# --images
# --------------------------------------------------------------------------- #


def _write_pdf_with_image(path: Path) -> None:
    """Two pages: one carrying a figure, one text only."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 64, 64))
    pix.set_rect(pix.irect, (20, 90, 160))
    png = pix.tobytes("png")

    doc = fitz.open()
    try:
        first = doc.new_page()
        first.insert_text((72, 720), "Figure 1: pump schematic", fontsize=11)
        first.insert_image(fitz.Rect(100, 150, 300, 350), stream=png)
        second = doc.new_page()
        second.insert_text(
            (72, 72), "Torque table and tightening procedure.", fontsize=11
        )
        doc.save(str(path))
    finally:
        doc.close()


@pytest.fixture
def pdf_with_image(tmp_path: Path) -> Path:
    path = tmp_path / "manual-with-figure.pdf"
    _write_pdf_with_image(path)
    return path


def test_images_json_reports_page_id_box_and_size(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(pdf_with_image), "--json", "--images"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"pages", "images"}
    assert len(payload["images"]) == 1
    image = payload["images"][0]
    assert image["id"] == "p1-i0"
    assert image["page"] == 1
    assert image["format"] == "png"
    assert image["bytes"] > 0
    assert image["bbox"] == pytest.approx([100.0, 150.0, 300.0, 350.0], abs=0.5)
    assert image["width"] == pytest.approx(200.0, abs=0.5)


def test_images_json_never_emits_the_bytes_themselves(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This CLI is meant to stay pipeable; a base64 blob would end that."""
    main([str(pdf_with_image), "--json", "--images"])

    image = json.loads(capsys.readouterr().out)["images"][0]
    assert "data" not in image
    assert isinstance(image["bytes"], int)


def test_json_without_images_keeps_the_old_shape(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Existing consumers parse a bare array of page records."""
    main([str(pdf_with_image), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    assert set(payload[0]) == {"page", "chunks"}


def test_human_readable_images_section(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(pdf_with_image), "--images"]) == 0

    out = capsys.readouterr().out
    assert "--- 1 image(s) ---" in out
    assert "[p1-i0]" in out
    assert "png" in out


def test_pages_filter_applies_to_images_too(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Otherwise the two halves of one report describe different documents."""
    assert main([str(pdf_with_image), "--json", "--images", "--pages", "2"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["images"] == []
    assert [p["page"] for p in payload["pages"]] == [2]


def test_min_image_size_filters_out_a_figure(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main([str(pdf_with_image), "--json", "--images", "--min-image-size", "500"])
        == 0
    )

    assert json.loads(capsys.readouterr().out)["images"] == []


def test_a_negative_min_image_size_exits_nonzero(
    pdf_with_image: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(pdf_with_image), "--images", "--min-image-size", "-1"]) != 0
    assert "min_size" in capsys.readouterr().err


def test_a_text_only_pdf_reports_no_images(
    pdf_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(pdf_path), "--json", "--images"]) == 0

    assert json.loads(capsys.readouterr().out)["images"] == []
