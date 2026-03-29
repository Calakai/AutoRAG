"""Core processing pipeline: PDF → text extraction → chunk → metadata-enriched results."""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from autorag.config import ProcessingConfig

logger = logging.getLogger(__name__)


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
    docling_version: str
    config: ProcessingConfig


class ProcessingCancelledError(Exception):
    pass


ProgressCallback = Optional[Callable[[str, float], None]]


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return f"sha256:{h.hexdigest()}"


def _report(callback: ProgressCallback, message: str, fraction: float) -> None:
    if callback:
        callback(message, fraction)


def _clean_text(text: str) -> str:
    """Remove common PDF extraction artifacts from raw text."""
    text = html.unescape(text)
    # Strip zero-width and other invisible unicode characters
    text = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad]", "", text)

    lines = text.split("\n")
    cleaned: list[str] = []
    prev_line = ""

    for line in lines:
        stripped = line.strip()

        # Collapse empty lines
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        # Skip duplicate consecutive lines
        if stripped == prev_line:
            continue

        # Skip lines that are just page numbers (1-4 digits)
        if re.match(r"^\d{1,4}$", stripped):
            continue

        # Skip lines that are only digits and spaces (DRM IDs, stray page numbers)
        if re.match(r"^[\d\s]+$", stripped) and len(stripped) > 2:
            continue

        # Skip lines that are just punctuation or very short non-words
        if len(stripped) <= 4 and sum(1 for c in stripped if c.isalpha()) <= 2:
            continue

        # Skip watermark lines containing @ signs (emails, DRM tags)
        if "@" in stripped and len(stripped.split()) > 2:
            continue

        # Check for DRM junk patterns using word-length analysis
        words = stripped.split()
        if len(words) >= 3:
            short_words = sum(1 for w in words if len(w.strip(".,;:!?")) <= 2)
            if short_words / len(words) > 0.5:
                continue

        # Skip lines with very low alphabetic ratio (gibberish)
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 8 and alpha_chars / len(stripped) < 0.5:
            continue

        # Strip trailing DRM junk appended to otherwise good lines
        # Pattern: real text followed by gibberish with lots of single-char words
        stripped = _strip_trailing_junk(stripped)
        if not stripped:
            continue

        cleaned.append(stripped)
        prev_line = stripped

    return "\n".join(cleaned)


def _strip_trailing_junk(line: str) -> str:
    """Remove DRM watermark junk appended to the end of a real text line."""
    words = line.split()
    if len(words) < 5:
        return line

    # Find the rightmost position where a "real" word exists (3+ alpha chars)
    # then check if everything after it is junk
    last_real = -1
    for i, w in enumerate(words):
        clean_w = w.strip(".,;:!?@()[]")
        if len(clean_w) >= 3 and sum(1 for c in clean_w if c.isalpha()) >= 3:
            last_real = i

    if last_real >= 2 and last_real < len(words) - 2:
        tail = words[last_real + 1:]
        short = sum(1 for w in tail if len(w.strip(".,;:!?@")) <= 2)
        if len(tail) >= 2 and short / len(tail) > 0.5:
            trimmed = " ".join(words[: last_real + 1]).rstrip(" ,;:")
            if trimmed:
                return trimmed

    return line


def _page_needs_ocr(text: str) -> bool:
    """Check if a page's extracted text is garbage — likely a scanned page.

    Heuristic: count words that are 3+ alpha characters (real words).
    Scanned pages produce noise characters but very few real words.
    """
    words = re.findall(r"[a-zA-Z]{3,}", text)
    return len(words) < 10


def _ocr_page(page, ocr_engine) -> str:
    """Render a PDF page to image and run OCR."""
    import io

    from PIL import Image

    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        result = ocr_engine(img)
        if result and result.txts:
            return " ".join(result.txts)
        return ""
    finally:
        img.close()


# Formats that pymupdf handles natively
_PYMUPDF_EXTENSIONS = {".pdf", ".epub", ".xps", ".oxps", ".cbz", ".fb2", ".mobi"}


def _extract_text_pymupdf(file_path: Path, progress_callback: ProgressCallback, cancel_check) -> tuple[list[str], int, bool]:
    """Extract text using PyMuPDF (PDF, EPUB, MOBI, XPS, FB2, CBZ) with auto-OCR."""
    import pymupdf

    doc = pymupdf.open(str(file_path))
    total_pages = len(doc)
    page_texts: list[str] = []
    ocr_engine = None
    ocr_count = 0

    for i in range(total_pages):
        if cancel_check and cancel_check():
            doc.close()
            raise ProcessingCancelledError()

        page = doc[i]
        text = page.get_text("text")

        if _page_needs_ocr(text):
            if ocr_engine is None:
                _report(progress_callback, "Loading OCR engine (scanned pages detected)...", 0.10)
                from rapidocr import RapidOCR
                ocr_engine = RapidOCR()
                logger.info("OCR engine loaded (scanned pages detected)")

            text = _ocr_page(page, ocr_engine)
            ocr_count += 1
            if (i + 1) % 5 == 0 or i == total_pages - 1:
                frac = 0.10 + 0.30 * (i + 1) / total_pages
                _report(progress_callback, f"OCR page {i + 1}/{total_pages} (scanned)", frac)
        else:
            if (i + 1) % 20 == 0 or i == total_pages - 1:
                frac = 0.10 + 0.30 * (i + 1) / total_pages
                _report(progress_callback, f"Extracting text... page {i + 1}/{total_pages}", frac)

        page_texts.append(_clean_text(text))

    if ocr_count > 0:
        logger.info("OCR was used on %d of %d pages", ocr_count, total_pages)

    doc.close()
    return page_texts, total_pages, ocr_count > 0


def _extract_text_docx(file_path: Path, progress_callback: ProgressCallback) -> tuple[list[str], int, bool]:
    """Extract text from DOCX using python-docx, preserving heading structure."""
    import docx

    _report(progress_callback, "Reading DOCX...", 0.15)
    doc = docx.Document(str(file_path))

    # Group paragraphs into logical "pages" (~3000 chars each for chunking consistency)
    page_texts: list[str] = []
    current_page: list[str] = []
    current_len = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name if para.style else "Normal"
        if style.startswith("Heading"):
            # Start a new page on major headings if current page has content
            if current_page and current_len > 500:
                page_texts.append("\n".join(current_page))
                current_page = []
                current_len = 0

        current_page.append(text)
        current_len += len(text)

        # Split on ~3000 char boundaries
        if current_len > 3000:
            page_texts.append("\n".join(current_page))
            current_page = []
            current_len = 0

    if current_page:
        page_texts.append("\n".join(current_page))

    _report(progress_callback, f"Read {len(page_texts)} sections from DOCX", 0.40)
    return [_clean_text(t) for t in page_texts], len(page_texts), False


def _extract_text_plain(file_path: Path, progress_callback: ProgressCallback) -> tuple[list[str], int, bool]:
    """Extract text from plain text or markdown files."""
    _report(progress_callback, "Reading text file...", 0.15)

    text = file_path.read_text(encoding="utf-8", errors="replace")

    # Split into logical pages by double newlines or ~3000 char boundaries
    sections = re.split(r"\n{3,}", text)
    page_texts: list[str] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Further split very long sections
        while len(section) > 3000:
            cut = section.rfind("\n", 0, 3000)
            if cut < 500:
                cut = 3000
            page_texts.append(section[:cut])
            section = section[cut:].strip()
        if section:
            page_texts.append(section)

    _report(progress_callback, f"Read {len(page_texts)} sections", 0.40)
    return [_clean_text(t) for t in page_texts], len(page_texts), False


def _extract_text(file_path: Path, progress_callback: ProgressCallback, cancel_check) -> tuple[list[str], int, bool]:
    """Route to the appropriate extractor based on file extension."""
    ext = file_path.suffix.lower()

    if ext in _PYMUPDF_EXTENSIONS:
        return _extract_text_pymupdf(file_path, progress_callback, cancel_check)
    elif ext == ".docx":
        return _extract_text_docx(file_path, progress_callback)
    elif ext in {".txt", ".md", ".text", ".rst"}:
        return _extract_text_plain(file_path, progress_callback)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _build_docling_document(
    name: str,
    page_texts: list[str],
    total_pages: int,
) -> "DoclingDocument":
    """Build a DoclingDocument from extracted page texts for chunking."""
    from docling_core.types.doc.base import BoundingBox
    from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name=name)
    page_bbox = BoundingBox(l=0, t=0, r=612, b=792)

    for page_no in range(total_pages):
        doc.add_page(page_no=page_no + 1, size={"width": 612, "height": 792})

    for page_no, text in enumerate(page_texts):
        if not text.strip():
            continue

        prov = ProvenanceItem(page_no=page_no + 1, bbox=page_bbox, charspan=(0, len(text)))
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            if _looks_like_heading(line):
                num_match = re.match(r"^(\d+(?:\.\d+)*)", line)
                level = min(num_match.group(1).count(".") + 1, 4) if num_match else 1
                doc.add_heading(text=line, level=level, prov=prov)
            else:
                # Accumulate consecutive non-empty lines into a paragraph
                para_lines = [line]
                while i + 1 < len(lines) and lines[i + 1].strip() and not _looks_like_heading(lines[i + 1].strip()):
                    i += 1
                    para_lines.append(lines[i].strip())
                doc.add_text(
                    text=" ".join(para_lines),
                    label=DocItemLabel.PARAGRAPH,
                    prov=prov,
                )
            i += 1

    return doc


def _looks_like_heading(line: str) -> bool:
    return (
        len(line) < 120
        and not line.endswith(".")
        and (
            line.istitle()
            or line.isupper()
            or re.match(r"^(Chapter|Part|Section|Appendix)\s", line, re.IGNORECASE) is not None
            or re.match(r"^\d+(\.\d+)*\s+\S", line) is not None
        )
    )


def process_document(
    file_path: Path,
    config: ProcessingConfig,
    progress_callback: ProgressCallback = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> PipelineResult:
    """Process a document: extract text, chunk with Docling's chunker.

    Supports PDF, EPUB, DOCX, TXT, MD, MOBI, XPS, FB2, CBZ.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    supported = _PYMUPDF_EXTENSIONS | {".docx", ".txt", ".md", ".text", ".rst"}
    if ext not in supported:
        raise ValueError(f"Unsupported format: {ext}")

    start_time = time.monotonic()
    filename_stem = file_path.stem

    # Report immediately so GUI shows activity before heavy imports
    _report(progress_callback, "Loading chunker (first time may be slow)...", 0.01)

    # Lazy imports — transformers tokenizer takes a few seconds on first load
    from docling_core.transforms.chunker import HierarchicalChunker, HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

    # Compute source hash
    _report(progress_callback, "Computing file hash...", 0.05)
    source_hash = _compute_sha256(file_path)

    if cancel_check and cancel_check():
        raise ProcessingCancelledError()

    # Extract text using the appropriate engine for this file type
    _report(progress_callback, "Extracting text...", 0.10)
    logger.info("Processing %s (%s)", file_path.name, ext)
    page_texts, total_pages, ocr_was_used = _extract_text(file_path, progress_callback, cancel_check)
    logger.info("Extracted text from %d pages", total_pages)

    if cancel_check and cancel_check():
        raise ProcessingCancelledError()

    # Build DoclingDocument from extracted text
    _report(progress_callback, "Building document structure...", 0.42)
    doc = _build_docling_document(filename_stem, page_texts, total_pages)

    # Export full markdown
    _report(progress_callback, "Exporting markdown...", 0.45)
    markdown = doc.export_to_markdown()

    if cancel_check and cancel_check():
        raise ProcessingCancelledError()

    # Set up tokenizer and chunker
    _report(progress_callback, "Setting up chunker...", 0.48)
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=config.chunking.max_tokens,
    )

    if config.chunking.strategy == "hierarchical":
        chunker = HierarchicalChunker(tokenizer=tokenizer)
    else:
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)

    # Chunk the document
    _report(progress_callback, "Chunking document...", 0.50)
    raw_chunks = list(chunker.chunk(doc))
    total_chunks = len(raw_chunks)
    logger.info("Produced %d chunks", total_chunks)

    if cancel_check and cancel_check():
        raise ProcessingCancelledError()

    # Build enriched chunk results
    _report(progress_callback, "Enriching chunk metadata...", 0.70)
    now = datetime.now(timezone.utc).isoformat()
    chunks: list[ChunkResult] = []
    pad = len(str(total_chunks))

    for i, chunk in enumerate(raw_chunks):
        # Extract page numbers from doc_items
        pages: set[int] = set()
        element_types: set[str] = set()
        for doc_item in chunk.meta.doc_items:
            if hasattr(doc_item, "page_no") and doc_item.page_no is not None:
                pages.add(doc_item.page_no)
            if hasattr(doc_item, "prov"):
                for prov in doc_item.prov:
                    if hasattr(prov, "page_no") and prov.page_no is not None:
                        pages.add(prov.page_no)
            if hasattr(doc_item, "label"):
                label = doc_item.label
                element_types.add(label.value if hasattr(label, "value") else str(label))

        sorted_pages = sorted(pages)
        page_start = sorted_pages[0] if sorted_pages else None
        page_end = sorted_pages[-1] if sorted_pages else None

        heading_path = chunk.meta.headings if chunk.meta.headings else []
        section_title = heading_path[-1] if heading_path else ""
        token_count = tokenizer.count_tokens(chunk.text)
        chunk_id = f"{filename_stem}_chunk_{i + 1:0{pad}d}"

        chunks.append(ChunkResult(
            chunk_id=chunk_id,
            text=chunk.text,
            token_count=token_count,
            metadata={
                "source_file": file_path.name,
                "page_start": page_start,
                "page_end": page_end,
                "section_title": section_title,
                "heading_path": heading_path,
                "element_types": sorted(element_types),
                "chunk_index": i + 1,
                "total_chunks": total_chunks,
                "custom_tags": list(config.metadata.custom_tags),
                "created_at": now,
            },
        ))

        if (i + 1) % 50 == 0 or i == total_chunks - 1:
            chunk_fraction = 0.70 + (0.20 * (i + 1) / total_chunks)
            _report(progress_callback, f"Processing chunk {i + 1}/{total_chunks}...", chunk_fraction)

    # Tables — not available in fast path (needs ML model)
    tables: list[dict] = []

    # Get docling version
    try:
        docling_version = importlib.metadata.version("docling")
    except Exception:
        docling_version = "unknown"

    elapsed = time.monotonic() - start_time
    _report(progress_callback, "Done!", 1.0)

    return PipelineResult(
        source_file=file_path.name,
        source_hash=source_hash,
        total_pages=total_pages,
        total_chunks=total_chunks,
        chunks=chunks,
        markdown=markdown,
        tables=tables,
        processing_time_seconds=round(elapsed, 2),
        ocr_used=ocr_was_used,
        docling_version=docling_version,
        config=config,
    )


# Backward-compatible alias
process_pdf = process_document
