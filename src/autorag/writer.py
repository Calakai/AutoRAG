"""Output writer: writes pipeline results to a structured output folder."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from autorag import __version__
from autorag.config import ProcessingConfig
from autorag.pipeline import PipelineResult


def write_output(result: PipelineResult, output_dir: Path) -> Path:
    """Write all output files for a processed document.

    Args:
        result: The pipeline result containing chunks, markdown, tables, etc.
        output_dir: Base output directory. A subdirectory named after the source file is created.

    Returns:
        Path to the document output directory.
    """
    doc_dir = output_dir / Path(result.source_file).stem
    chunks_dir = doc_dir / "chunks"
    markdown_dir = doc_dir / "markdown"
    tables_dir = doc_dir / "tables"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    # Write individual chunk JSON files
    pad_width = len(str(result.total_chunks)) if result.total_chunks > 0 else 3
    for i, chunk in enumerate(result.chunks):
        chunk_dict = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata,
        }
        chunk_path = chunks_dir / f"chunk_{i + 1:0{pad_width}d}.json"
        chunk_path.write_text(
            json.dumps(chunk_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Write combined JSONL
    jsonl_path = doc_dir / "chunks_combined.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in result.chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")

    # Write full markdown
    md_path = markdown_dir / "full_document.md"
    md_path.write_text(result.markdown, encoding="utf-8")

    # Write tables (if any)
    if result.tables:
        tables_dir.mkdir(parents=True, exist_ok=True)
        for table in result.tables:
            table_path = tables_dir / f"{table['table_id']}.md"
            table_path.write_text(table["markdown"], encoding="utf-8")

    # Write manifest
    manifest = {
        "autorag_version": __version__,
        "source_file": result.source_file,
        "source_hash": result.source_hash,
        "total_pages": result.total_pages,
        "total_chunks": result.total_chunks,
        "chunking_strategy": result.config.chunking.strategy,
        "max_tokens": result.config.chunking.max_tokens,
        "overlap_tokens": result.config.chunking.overlap_tokens,
        "ocr_used": result.ocr_used,
        "scanned_pages": result.scanned_pages,
        "ocr_pages_skipped": result.ocr_pages_skipped,
        "processing_time_seconds": result.processing_time_seconds,
        "chunker": result.chunker,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = doc_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write config snapshot
    result.config.to_toml(doc_dir / "config_snapshot.toml")

    return doc_dir
