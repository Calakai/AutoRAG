"""Tests for config, pipeline, and writer modules."""

import json
from pathlib import Path

import pytest

from autorag.config import ProcessingConfig
from autorag.pipeline import ChunkResult, PipelineResult, process_document
from autorag.writer import write_output


# --- Config Tests ---


def test_config_defaults():
    cfg = ProcessingConfig()
    assert cfg.chunking.strategy == "hybrid"
    assert cfg.chunking.max_tokens == 512
    assert cfg.chunking.overlap_tokens == 64
    assert cfg.parsing.ocr_enabled is False
    assert cfg.general.output_dir == "./output"
    assert cfg.metadata.custom_tags == []


def test_config_toml_roundtrip(tmp_path: Path):
    cfg = ProcessingConfig()
    cfg.chunking.max_tokens = 256
    cfg.parsing.ocr_enabled = True
    cfg.metadata.custom_tags = ["test", "demo"]

    toml_path = tmp_path / "test.toml"
    cfg.to_toml(toml_path)

    loaded = ProcessingConfig.from_toml(toml_path)
    assert loaded.chunking.max_tokens == 256
    assert loaded.parsing.ocr_enabled is True
    assert loaded.metadata.custom_tags == ["test", "demo"]
    assert loaded == cfg


def test_config_load_defaults():
    cfg = ProcessingConfig.load_defaults()
    assert cfg.chunking.strategy == "hybrid"
    assert cfg.chunking.max_tokens == 512
    assert cfg.general.log_level == "INFO"


def test_config_ignores_unknown_keys(tmp_path: Path):
    # Snapshots from v0.1 carry ocr_engine; loading them must not crash
    old = tmp_path / "old.toml"
    old.write_text('[parsing]\nocr_engine = "easyocr"\nmax_pages = 3\n[future]\nx = 1\n')
    assert ProcessingConfig.from_toml(old).parsing.max_pages == 3


# --- Pipeline Tests (offline: no tokenizer download) ---

FIXTURES = Path(__file__).parent / "fixtures"


def test_process_markdown_fixture():
    result = process_document(FIXTURES / "test.md", ProcessingConfig())
    assert result.total_chunks >= 1
    first = result.chunks[0]
    assert first.chunk_id == "test_chunk_001"
    assert first.metadata["heading_path"][0] == "Test Markdown"
    assert first.metadata["page_start"] is None  # markdown has no pages
    assert result.markdown.startswith("# Test Markdown")
    assert result.source_hash.startswith("sha256:")
    assert result.chunker.startswith("autorag-hybrid/")


def test_process_pdf_reports_pages(tmp_path: Path):
    from tests.helpers import LOREM, make_pdf

    pdf = make_pdf(tmp_path / "guide.pdf", [f"Getting Started\n{LOREM}", f"Troubleshooting\n{LOREM}"])
    config = ProcessingConfig()
    config.chunking.strategy = "hierarchical"
    result = process_document(pdf, config)
    assert result.total_pages == 2
    spans = [(c.metadata["page_start"], c.metadata["page_end"], c.metadata["section_title"]) for c in result.chunks]
    assert spans == [(1, 1, "Getting Started"), (2, 2, "Troubleshooting")]

    # hybrid merges the two small sections but keeps their headings in the text
    merged = process_document(pdf, ProcessingConfig()).chunks
    assert len(merged) == 1 and merged[0].metadata["page_end"] == 2
    assert "Getting Started" in merged[0].text and "Troubleshooting" in merged[0].text


def test_process_rejects_unsupported(tmp_path: Path):
    bad = tmp_path / "book.mobi"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        process_document(bad, ProcessingConfig())


# --- Writer Tests ---


def _make_mock_result(num_chunks: int = 3) -> PipelineResult:
    chunks = [
        ChunkResult(
            chunk_id=f"test_chunk_{i + 1:03d}",
            text=f"This is chunk {i + 1} content.",
            token_count=10,
            metadata={
                "source_file": "test.pdf",
                "page_start": 1,
                "page_end": 1,
                "section_title": "Introduction",
                "heading_path": ["Introduction"],
                "element_types": ["paragraph"],
                "chunk_index": i + 1,
                "total_chunks": num_chunks,
                "custom_tags": [],
                "created_at": "2026-03-28T00:00:00+00:00",
            },
        )
        for i in range(num_chunks)
    ]
    return PipelineResult(
        source_file="test.pdf",
        source_hash="sha256:abc123",
        total_pages=5,
        total_chunks=num_chunks,
        chunks=chunks,
        markdown="# Test Document\n\nSome content here.",
        tables=[
            {"table_id": "table_001", "markdown": "| A | B |\n|---|---|\n| 1 | 2 |"}
        ],
        processing_time_seconds=1.23,
        ocr_used=False,
        chunker="autorag-hybrid/test",
        config=ProcessingConfig(),
    )


def test_writer_output_structure(tmp_path: Path):
    result = _make_mock_result()
    output_dir = tmp_path / "output"
    doc_dir = write_output(result, output_dir)

    assert doc_dir.exists()
    assert (doc_dir / "manifest.json").exists()
    assert (doc_dir / "chunks_combined.jsonl").exists()
    assert (doc_dir / "markdown" / "full_document.md").exists()
    assert (doc_dir / "config_snapshot.toml").exists()
    assert (doc_dir / "tables" / "table_001.md").exists()

    # Verify chunk files
    chunk_files = sorted((doc_dir / "chunks").glob("chunk_*.json"))
    assert len(chunk_files) == 3

    # Verify chunk JSON schema
    with open(chunk_files[0], encoding="utf-8") as f:
        chunk_data = json.load(f)
    assert "chunk_id" in chunk_data
    assert "text" in chunk_data
    assert "token_count" in chunk_data
    assert "metadata" in chunk_data
    assert chunk_data["metadata"]["source_file"] == "test.pdf"

    # Verify JSONL line count
    with open(doc_dir / "chunks_combined.jsonl", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 3

    # Verify manifest
    with open(doc_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["total_chunks"] == 3
    assert manifest["total_pages"] == 5
    assert manifest["source_hash"] == "sha256:abc123"
    assert manifest["chunking_strategy"] == "hybrid"
    assert manifest["chunker"] == "autorag-hybrid/test"
    assert "docling_version" not in manifest

    # Verify markdown
    md_content = (doc_dir / "markdown" / "full_document.md").read_text(encoding="utf-8")
    assert "Test Document" in md_content


def test_writer_no_tables(tmp_path: Path):
    result = _make_mock_result()
    result.tables = []
    output_dir = tmp_path / "output"
    doc_dir = write_output(result, output_dir)
    assert not (doc_dir / "tables").exists()
