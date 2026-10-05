"""Text extraction: document file → per-page text.

Every format here is read with a permissively licensed library or the stdlib.
PDF goes through pdfium (pypdfium2, Apache-2.0/BSD); epub, fb2 and xps/oxps are
zip/XML underneath and are parsed with the stdlib. PyMuPDF (AGPL) is gone — it
blocked MIT-licensing this project and embedding it in MonkChat, which made the
same swap. The paged-document seam, the ebook parsers and the OCR backend
selection are ported from MonkChat's importing.py, which itself ported AutoRAG's
original OCR recipe and text cleaners.

MOBI and CBZ are no longer supported: MOBI is a proprietary binary format with no
permissive reader, and CBZ is image-only (full-page OCR of comic lettering is
poor).
"""

from __future__ import annotations

import base64
import html
import io
import logging
import posixpath
import re
import threading
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

PAGED_EXTENSIONS = {".pdf", ".epub", ".fb2", ".xps", ".oxps"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".text", ".rst"}
HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = PAGED_EXTENSIONS | TEXT_EXTENSIONS | HTML_EXTENSIONS | {".docx"}

OCR_DPI = 200  # rasterization density (AutoRAG's proven default)
OCR_MAX_PAGES = 2000  # finite bound so a pathological document can't OCR forever
_OCR_MIN_ALPHA_RUNS = 10  # fewer runs of 3+ letters than this ⇒ treat the page as scanned

ProgressCallback = Callable[[str, float], None] | None


class UnsupportedFormatError(ValueError):
    """The file isn't a document format AutoRAG can read."""


class ProcessingCancelledError(Exception):
    """The caller's cancel_check asked processing to stop."""


@dataclass
class Extraction:
    """Extracted text plus an honest account of what couldn't be read.

    `pages` holds one string per page for paged formats. Page-less formats (docx,
    text, html) return a single entry and `paged=False`, so callers don't invent
    page numbers. `ocr_pages_skipped` counts scanned pages left unread — past the
    page cap, or with no OCR engine installed.
    """

    pages: list[str]
    paged: bool
    scanned_pages: int = 0
    ocr_pages_read: int = 0
    ocr_pages_skipped: int = 0

    @property
    def ocr_used(self) -> bool:
        return self.ocr_pages_read > 0


def _report(callback: ProgressCallback, message: str, fraction: float) -> None:
    if callback:
        callback(message, fraction)


# --- OCR --------------------------------------------------------------------------
# Two optional engines, preferred in order: Apple Vision via ocrmac (macOS, native)
# and RapidOCR v3 (portable, onnxruntime). Built lazily on first scanned page and
# cached. With neither installed, scanned pages are counted as skipped rather than
# failing the document.

_ocr_backend = None
_ocr_lock = threading.Lock()


def _page_needs_ocr(text: str) -> bool:
    """A page with almost no real words is a scanned/image page worth OCR'ing."""
    return len(re.findall(r"[a-zA-Z]{3,}", text or "")) < _OCR_MIN_ALPHA_RUNS


def ocr_available() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    for module in ("ocrmac", "rapidocr"):
        try:
            __import__(module)
            return True
        except ImportError:
            continue
    return False


def _build_ocr_backend() -> Callable:
    try:
        from ocrmac import ocrmac

        def _vision(img) -> str:
            # One detected line per output line: the cleaners filter per line, so a
            # page-as-one-line would let a single junk token drop the whole page.
            anns = ocrmac.OCR(img, recognition_level="accurate").recognize()
            return "\n".join(a[0] for a in anns).strip()

        logger.info("OCR backend: Apple Vision (ocrmac)")
        return _vision
    except ImportError:
        pass

    import numpy as np
    from rapidocr import RapidOCR

    engine = RapidOCR()

    def _rapid(img) -> str:
        result = engine(np.array(img))
        txts = getattr(result, "txts", None)
        return "\n".join(txts).strip() if txts else ""

    logger.info("OCR backend: RapidOCR (onnxruntime)")
    return _rapid


def _get_ocr_backend() -> Callable:
    global _ocr_backend
    if _ocr_backend is None:
        with _ocr_lock:
            if _ocr_backend is None:
                _ocr_backend = _build_ocr_backend()
    return _ocr_backend


# --- Paged-document seam ------------------------------------------------------------
# Every page-oriented format presents the same surface — `page_count`,
# `doc[i].get_text()`, `doc[i].pil_images()`, `close()` — so the weak-page OCR
# machinery has one shape to reason about.


class _PdfiumPage:
    """Lazy view of one PDF page: opens the native page per call, so a 2,000-page
    book never holds 2,000 native handles."""

    def __init__(self, doc, index: int) -> None:
        self._doc = doc
        self._index = index

    def get_text(self) -> str:
        page = self._doc.get_page(self._index)
        try:
            textpage = page.get_textpage()
            try:
                return textpage.get_text_range().replace("\r\n", "\n")
            finally:
                textpage.close()
        finally:
            page.close()

    def has_images(self) -> bool:
        return True  # a PDF page is rendered whole, so there is always something to OCR

    def pil_images(self) -> list:
        page = self._doc.get_page(self._index)
        try:
            bitmap = page.render(scale=OCR_DPI / 72)
            try:
                return [bitmap.to_pil()]
            finally:
                bitmap.close()
        finally:
            page.close()


class _PdfiumDoc:
    def __init__(self, raw) -> None:
        self._raw = raw
        self.page_count = len(raw)

    def __getitem__(self, index: int) -> _PdfiumPage:
        return _PdfiumPage(self._raw, index)

    def close(self) -> None:
        self._raw.close()


class _EbookPage:
    def __init__(self, text: str, images: Callable[[], list], image_count: int = 0) -> None:
        self._text = text
        self._images = images
        self._image_count = image_count

    def get_text(self) -> str:
        return self._text

    def has_images(self) -> bool:
        return self._image_count > 0

    def pil_images(self) -> list:
        return self._images()


class _EbookDoc:
    def __init__(self, pages: list[_EbookPage], close: Callable[[], None]) -> None:
        self._pages = pages
        self._close = close
        self.page_count = len(pages)

    def __getitem__(self, index: int) -> _EbookPage:
        return self._pages[index]

    def close(self) -> None:
        self._close()


def _open_paged(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pypdfium2 as pdfium

        return _PdfiumDoc(pdfium.PdfDocument(str(path)))
    if ext == ".epub":
        return _open_epub(path)
    if ext == ".fb2":
        return _open_fb2(path)
    return _open_xps(path)


def page_count(path: Path) -> int | None:
    """Number of pages for a paged format, None for page-less formats."""
    if path.suffix.lower() not in PAGED_EXTENSIONS:
        return None
    doc = _open_paged(path)
    try:
        return doc.page_count
    finally:
        doc.close()


# --- Ebook / fixed-layout parsing (stdlib only) ---------------------------------------
# A "page" is the natural unit of each format: an epub spine chapter, an fb2 <body>,
# an xps fixed page.

_EBOOK_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp")


class _HtmlStripper(HTMLParser):
    """HTML → text, keeping block boundaries as newlines and headings as markdown."""

    _BLOCK = {"p", "div", "br", "li", "tr", "section", "article", "blockquote", "pre", "table"}
    _SKIP = {"script", "style", "head", "title", "noscript"}
    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._HEADINGS:
            self._parts.append("\n\n" + "#" * self._HEADINGS[tag] + " ")
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._HEADINGS or tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in raw.split("\n")]
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class _ChapterHtml(_HtmlStripper):
    """Also records embedded images, so an image-only chapter can still be OCR'd."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        super().handle_starttag(tag, attrs)
        if tag in {"img", "image"}:
            for name, value in attrs:
                if name in {"src", "href", "xlink:href"} and value:
                    self.images.append(value)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_xml(data: bytes) -> ET.Element:
    """ET.fromstring with a DTD guard: entity expansion (billion laughs) rides in on
    <!DOCTYPE>, and no ebook dialect parsed here legitimately carries one."""
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise UnsupportedFormatError("XML with a DTD is not accepted.")
    return ET.fromstring(data)


def _resolve_zip_href(base: str, href: str) -> str:
    return posixpath.normpath(posixpath.join(base, unquote(href.split("#", 1)[0])))


def _zip_image_loader(zf: zipfile.ZipFile, names: list[str]) -> Callable[[], list]:
    def load() -> list:
        from PIL import Image

        images = []
        for name in names:
            try:
                img = Image.open(io.BytesIO(zf.read(name)))
                img.load()
                images.append(img)
            except Exception:
                logger.debug("unreadable embedded image %s", name, exc_info=True)
        return images

    return load


def _epub_spine(zf: zipfile.ZipFile, names: set[str]) -> list[str]:
    """Spine-ordered content documents, falling back to archive order when the
    package metadata is missing or malformed."""
    try:
        container = _parse_xml(zf.read("META-INF/container.xml"))
        opf_path = next(
            el.attrib["full-path"]
            for el in container.iter()
            if _local(el.tag) == "rootfile" and "full-path" in el.attrib
        )
        opf = _parse_xml(zf.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {
            el.attrib["id"]: el.attrib["href"]
            for el in opf.iter()
            if _local(el.tag) == "item" and "id" in el.attrib and "href" in el.attrib
        }
        spine = [
            _resolve_zip_href(base, manifest[el.attrib["idref"]])
            for el in opf.iter()
            if _local(el.tag) == "itemref" and el.attrib.get("idref") in manifest
        ]
        spine = [s for s in spine if s in names]
        if spine:
            return spine
    except Exception:
        logger.debug("epub package metadata unreadable; using archive order", exc_info=True)
    return [n for n in zf.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]


def _open_epub(path: Path) -> _EbookDoc:
    zf = zipfile.ZipFile(path)
    names = set(zf.namelist())
    pages: list[_EbookPage] = []
    for doc_name in _epub_spine(zf, names):
        parser = _ChapterHtml()
        parser.feed(zf.read(doc_name).decode("utf-8", errors="replace"))
        base = posixpath.dirname(doc_name)
        images = [
            resolved
            for src in parser.images
            if (resolved := _resolve_zip_href(base, src)) in names
            and resolved.lower().endswith(_EBOOK_IMAGE_EXTS)
        ]
        pages.append(_EbookPage(parser.text(), _zip_image_loader(zf, images), len(images)))
    return _EbookDoc(pages, zf.close)


def _open_fb2(path: Path) -> _EbookDoc:
    root = _parse_xml(path.read_bytes())
    binaries = [el.text for el in root.iter() if _local(el.tag) == "binary" and el.text]

    def load_binaries() -> list:
        from PIL import Image

        images = []
        for b64 in binaries:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(b64)))
                img.load()
                images.append(img)
            except Exception:
                logger.debug("unreadable fb2 binary", exc_info=True)
        return images

    pages = [
        _EbookPage("\n".join(t.strip() for t in body.itertext() if t.strip()), load_binaries, len(binaries))
        for body in root
        if _local(body.tag) == "body"
    ]
    return _EbookDoc(pages, lambda: None)


def _natural_key(name: str) -> list:
    return [(0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", name)]


def _open_xps(path: Path) -> _EbookDoc:
    zf = zipfile.ZipFile(path)
    pages: list[_EbookPage] = []
    for name in sorted((n for n in zf.namelist() if n.lower().endswith(".fpage")), key=_natural_key):
        try:
            root = _parse_xml(zf.read(name))
        except (ET.ParseError, UnsupportedFormatError):
            pages.append(_EbookPage("", lambda: []))
            continue
        runs = [
            el.attrib["UnicodeString"]
            for el in root.iter()
            if _local(el.tag) == "Glyphs" and el.attrib.get("UnicodeString")
        ]
        pages.append(_EbookPage("\n".join(runs), lambda: []))
    return _EbookDoc(pages, zf.close)


# --- Extraction hygiene (paged/OCR output only) -------------------------------------
# Strips the junk page-oriented sources accumulate: page numbers, DRM/watermark tags,
# zero-width unicode, duplicated lines, OCR gibberish. Deliberately NOT applied to
# text/markdown/docx/html, which need exact fidelity.


def clean_text(text: str) -> str:
    """Remove common PDF/ebook extraction artifacts from raw text."""
    text = html.unescape(text)
    text = re.sub(r"[​‌‍‎‏﻿­]", "", text)

    cleaned: list[str] = []
    prev_line = ""
    for line in text.split("\n"):
        stripped = line.strip()

        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if stripped == prev_line:
            continue
        # Bare page numbers, and digit-only lines (DRM ids, stray numbering)
        if re.match(r"^\d{1,4}$", stripped):
            continue
        if re.match(r"^[\d\s]+$", stripped) and len(stripped) > 2:
            continue
        # Punctuation or very short non-words
        if len(stripped) <= 4 and sum(1 for c in stripped if c.isalpha()) <= 2:
            continue
        # Watermark lines carrying an email/DRM tag
        if "@" in stripped and len(stripped.split()) > 2:
            continue
        # DRM junk: mostly one- and two-letter "words"
        words = stripped.split()
        if len(words) >= 3:
            short_words = sum(1 for w in words if len(w.strip(".,;:!?")) <= 2)
            if short_words / len(words) > 0.5:
                continue
        # Low alphabetic ratio (gibberish)
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 8 and alpha_chars / len(stripped) < 0.5:
            continue

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

    last_real = -1
    for i, w in enumerate(words):
        clean_w = w.strip(".,;:!?@()[]")
        if len(clean_w) >= 3 and sum(1 for c in clean_w if c.isalpha()) >= 3:
            last_real = i

    if 2 <= last_real < len(words) - 2:
        tail = words[last_real + 1 :]
        short = sum(1 for w in tail if len(w.strip(".,;:!?@")) <= 2)
        if len(tail) >= 2 and short / len(tail) > 0.5:
            trimmed = " ".join(words[: last_real + 1]).rstrip(" ,;:")
            if trimmed:
                return trimmed
    return line


# --- Per-format extractors ------------------------------------------------------------


def _extract_paged(
    path: Path,
    progress: ProgressCallback,
    cancel_check: Callable[[], bool] | None,
    max_ocr_pages: int,
) -> Extraction:
    doc = _open_paged(path)
    try:
        total = doc.page_count
        pages: list[str] = []
        for i in range(total):
            if cancel_check and cancel_check():
                raise ProcessingCancelledError()
            pages.append(doc[i].get_text() or "")
            if (i + 1) % 20 == 0 or i == total - 1:
                _report(progress, f"Extracting text... page {i + 1}/{total}", 0.10 + 0.20 * (i + 1) / total)

        # Only pages with an image to read count as scanned: a short text-only
        # title page is not a scan, and OCR could add nothing to it.
        weak = [i for i, text in enumerate(pages) if _page_needs_ocr(text) and doc[i].has_images()]
        is_pdf = path.suffix.lower() == ".pdf"
        read = 0
        if weak and ocr_available():
            try:
                backend = _get_ocr_backend()
            except Exception as exc:  # a broken OCR stack must never sink the document
                logger.warning("OCR unavailable for %s: %s", path.name, exc)
                backend = None
            for count, i in enumerate(weak):
                if backend is None or count >= max_ocr_pages:
                    break
                if cancel_check and cancel_check():
                    raise ProcessingCancelledError()
                try:
                    text = "\n".join(t for t in (backend(img) for img in doc[i].pil_images()) if t).strip()
                    if text:
                        # A rendered PDF page already contains its text layer; an
                        # ebook page's images don't, so keep the page's own words.
                        pages[i] = text if is_pdf or not pages[i].strip() else f"{pages[i]}\n{text}"
                        read += 1
                except Exception as exc:
                    logger.warning("OCR failed on page %d of %s: %s", i + 1, path.name, exc)
                _report(progress, f"OCR page {count + 1}/{len(weak)} (scanned)", 0.30 + 0.10 * (count + 1) / len(weak))
        elif weak:
            logger.warning(
                "%s: %d scanned page(s) left unread — install the [ocr] extra to read them",
                path.name,
                len(weak),
            )
    finally:
        doc.close()

    return Extraction(
        pages=[clean_text(p) for p in pages],
        paged=True,
        scanned_pages=len(weak),
        ocr_pages_read=read,
        ocr_pages_skipped=len(weak) - read,
    )


def _extract_docx(path: Path) -> Extraction:
    """DOCX → markdown-ish text: headings keep their level as `#` prefixes so the
    chunker sees real structure instead of guessing from capitalization."""
    import docx

    document = docx.Document(str(path))
    lines: list[str] = []
    # iter_inner_content yields paragraphs and tables in document order, so a table
    # stays under the heading it appears beneath.
    for item in document.iter_inner_content():
        if hasattr(item, "rows"):
            for row in item.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
            lines.append("")
            continue
        text = item.text.strip()
        if not text:
            continue
        style = item.style.name if item.style is not None else ""
        match = re.match(r"Heading (\d)", style)
        if match:
            lines.append("#" * min(int(match.group(1)), 6) + " " + text)
        elif style == "Title":
            lines.append("# " + text)
        else:
            lines.append(text)
        lines.append("")
    return Extraction(pages=["\n".join(lines).strip()], paged=False)


def _extract_html(path: Path) -> Extraction:
    parser = _HtmlStripper()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return Extraction(pages=[parser.text()], paged=False)


def _extract_plain(path: Path) -> Extraction:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").lstrip("﻿")
    return Extraction(pages=[text], paged=False)


def extract(
    path: Path,
    progress: ProgressCallback = None,
    cancel_check: Callable[[], bool] | None = None,
    max_ocr_pages: int = OCR_MAX_PAGES,
) -> Extraction:
    """Extract text from a supported document, routing by file extension."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"Unsupported file format: {ext or '(none)'}")
    try:
        if ext in PAGED_EXTENSIONS:
            return _extract_paged(path, progress, cancel_check, max_ocr_pages)
        _report(progress, f"Reading {ext.lstrip('.').upper()}...", 0.15)
        if ext == ".docx":
            return _extract_docx(path)
        if ext in HTML_EXTENSIONS:
            return _extract_html(path)
        return _extract_plain(path)
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise UnsupportedFormatError(f"{path.name} is not a valid {ext} file: {exc}") from exc
