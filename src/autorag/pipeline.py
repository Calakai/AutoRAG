"""Core processing pipeline: document → text extraction → chunks with provenance."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autorag import __version__
from autorag.chunking import blocks_to_markdown, chunk_blocks, parse_blocks
from autorag.config import ProcessingConfig
from autorag.extract import SUPPORTED_EXTENSIONS, ProcessingCancelledError, extract

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkResult",
    "PipelineResult",
    "ProcessingCancelledError",
    "SUPPORTED_EXTENSIONS",
    "compute_sha256",
    "process_document",
]


@dataclass
class ChunkResult:
    chunk_id: str
    text: str
    token_count: int
    metadata: dict


@dataclass
class PipelineResult:
    source_file: str
    source_hash: str
    total_pages: int
    total_chunks: int
    chunks: list[ChunkResult]
    markdown: str
    tables: list[dict]
    processing_time_seconds: float
    ocr_used: bool
    chunker: str
    config: ProcessingConfig
    scanned_pages: int = 0
    ocr_pages_skipped: int = 0


ProgressCallback = Optional[Callable[[str, float], None]]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def _report(callback: ProgressCallback, message: str, fraction: float) -> None:
    if callback:
        callback(message, fraction)


def process_document(
    file_path: Path,
    config: ProcessingConfig,
    progress_callback: ProgressCallback = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    source_hash: str | None = None,
) -> PipelineResult:
    """Extract and chunk a document.

    Supports PDF, EPUB, FB2, XPS/OXPS, DOCX, HTML, TXT, MD and RST.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")

    start_time = time.monotonic()
    _report(progress_callback, "Computing file hash...", 0.05)
    source_hash = source_hash or compute_sha256(file_path)
    if cancel_check and cancel_check():
        raise ProcessingCancelledError()

    _report(progress_callback, "Extracting text...", 0.10)
    logger.info("Processing %s (%s)", file_path.name, ext)
    extraction = extract(file_path, progress=progress_callback, cancel_check=cancel_check)
    if cancel_check and cancel_check():
        raise ProcessingCancelledError()

    _report(progress_callback, "Finding document structure...", 0.45)
    blocks = parse_blocks(extraction.pages, extraction.paged)
    markdown = blocks_to_markdown(blocks)

    _report(progress_callback, "Chunking document...", 0.55)
    raw_chunks = chunk_blocks(
        blocks,
        max_tokens=config.chunking.max_tokens,
        overlap_tokens=config.chunking.overlap_tokens,
        strategy=config.chunking.strategy,
    )
    total_chunks = len(raw_chunks)
    logger.info("Produced %d chunks", total_chunks)

    now = datetime.now(timezone.utc).isoformat()
    pad = max(3, len(str(total_chunks)))
    stem = file_path.stem
    chunks = [
        ChunkResult(
            chunk_id=f"{stem}_chunk_{i + 1:0{pad}d}",
            text=chunk.text,
            token_count=chunk.token_count,
            metadata={
                "source_file": file_path.name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_title": chunk.heading_path[-1] if chunk.heading_path else "",
                "heading_path": chunk.heading_path,
                "element_types": chunk.element_types,
                "chunk_index": i + 1,
                "total_chunks": total_chunks,
                "custom_tags": list(config.metadata.custom_tags),
                "created_at": now,
            },
        )
        for i, chunk in enumerate(raw_chunks)
    ]

    elapsed = time.monotonic() - start_time
    _report(progress_callback, "Done!", 1.0)
    return PipelineResult(
        source_file=file_path.name,
        source_hash=source_hash,
        total_pages=len(extraction.pages) if extraction.paged else 0,
        total_chunks=total_chunks,
        chunks=chunks,
        markdown=markdown,
        tables=[],
        processing_time_seconds=round(elapsed, 2),
        ocr_used=extraction.ocr_used,
        chunker=f"autorag-{config.chunking.strategy}/{__version__}",
        config=config,
        scanned_pages=extraction.scanned_pages,
        ocr_pages_skipped=extraction.ocr_pages_skipped,
    )


# Backward-compatible alias
process_pdf = process_document
