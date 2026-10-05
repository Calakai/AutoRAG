"""Extraction for every supported format, using real files built on the fly."""

from pathlib import Path

import pytest

from autorag import extract as extract_mod
from autorag.extract import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFormatError,
    clean_text,
    extract,
    page_count,
)
from tests.helpers import LOREM, make_epub, make_fb2, make_pdf, make_xps

FIXTURES = Path(__file__).parent / "fixtures"


def test_pymupdf_formats_dropped_and_new_ones_added():
    assert ".mobi" not in SUPPORTED_EXTENSIONS and ".cbz" not in SUPPORTED_EXTENSIONS
    assert {".pdf", ".epub", ".docx", ".html", ".rst", ".oxps"} <= SUPPORTED_EXTENSIONS


def test_pdf_text_layer_per_page(tmp_path):
    pdf = make_pdf(tmp_path / "net.pdf", [f"Routing Basics\n{LOREM}", f"Switching\n{LOREM}"])
    result = extract(pdf)
    assert result.paged and len(result.pages) == 2
    assert "Routing Basics" in result.pages[0]
    assert "longest prefix wins" in result.pages[1]
    assert page_count(pdf) == 2
    assert result.scanned_pages == 0 and not result.ocr_used


def test_pdf_page_without_text_counts_as_scanned(tmp_path, monkeypatch):
    monkeypatch.setattr(extract_mod, "ocr_available", lambda: False)
    pdf = make_pdf(tmp_path / "scan.pdf", [LOREM, ""])
    result = extract(pdf)
    assert result.scanned_pages == 1
    assert result.ocr_pages_skipped == 1  # reported, not silently lost


def test_ocr_backend_fills_scanned_pages(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setattr(extract_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(extract_mod, "_get_ocr_backend", lambda: (lambda img: "Recovered words from a scanned page here"))
    pdf = make_pdf(tmp_path / "scan.pdf", [LOREM, ""])
    result = extract(pdf)
    assert result.ocr_pages_read == 1 and result.ocr_pages_skipped == 0 and result.ocr_used
    assert "Recovered words" in result.pages[1]


def test_short_ebook_page_keeps_its_text_and_only_image_pages_count(tmp_path, monkeypatch):
    import zipfile

    from tests.helpers import make_epub as _make

    epub = _make(tmp_path / "b.epub", [("Title", "By Cal"), ("Body", LOREM)])
    result = extract(epub)
    assert result.scanned_pages == 0  # a short text-only title page is not a scan

    # Add an illustration to the short chapter: OCR text is appended, not substituted
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    with zipfile.ZipFile(epub, "a") as zf:
        zf.writestr("OEBPS/text/pic.png", buf.getvalue())
    with zipfile.ZipFile(epub) as zf:
        entries = {n: zf.read(n) for n in zf.namelist()}
    entries["OEBPS/text/ch0.xhtml"] = entries["OEBPS/text/ch0.xhtml"].replace(b"</p>", b'</p><img src="pic.png"/>')
    with zipfile.ZipFile(epub, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    monkeypatch.setattr(extract_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(extract_mod, "_get_ocr_backend", lambda: (lambda img: "Map of the harbour"))
    result = extract(epub)
    assert result.scanned_pages == 1 and result.ocr_pages_read == 1
    assert "By Cal" in result.pages[0] and "Map of the harbour" in result.pages[0]


def test_epub_follows_spine_order(tmp_path):
    epub = make_epub(tmp_path / "book.epub", [("First Chapter", LOREM), ("Second Chapter", "Closing words.")])
    result = extract(epub)
    assert result.paged and len(result.pages) == 2
    assert result.pages[0].startswith("# First Chapter")
    assert "Second Chapter" in result.pages[1]
    assert "p{}" not in result.pages[0]  # <style> stripped


def test_fb2_and_xps(tmp_path):
    fb2 = extract(make_fb2(tmp_path / "book.fb2", LOREM))
    assert "Chapter One" in fb2.pages[0] and "routing tables" in fb2.pages[0]
    xps = extract(make_xps(tmp_path / "doc.xps", ["Page one has enough real words to count as text here", "Page two"]))
    assert xps.pages[0].startswith("Page one") and len(xps.pages) == 2


def test_xml_with_dtd_is_rejected(tmp_path):
    bomb = tmp_path / "bomb.fb2"
    bomb.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><FictionBook>&a;</FictionBook>')
    with pytest.raises(UnsupportedFormatError):
        extract(bomb)


def test_docx_headings_become_markdown(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Setup Guide", level=1)
    document.add_paragraph("Plug the router in.")
    document.add_heading("Advanced", level=2)
    document.add_paragraph("Edit the routing table.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Port"
    table.rows[0].cells[1].text = "22"
    document.add_heading("Appendix", level=2)
    document.add_paragraph("Extra notes.")
    path = tmp_path / "guide.docx"
    document.save(path)

    result = extract(path)
    assert not result.paged
    text = result.pages[0]
    assert "# Setup Guide" in text and "## Advanced" in text and "Port | 22" in text
    # The table stays under the heading it appears beneath, not at the end
    assert text.index("## Advanced") < text.index("Port | 22") < text.index("## Appendix")


def test_html_strips_scripts_and_keeps_headings(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<html><script>alert(1)</script><body><h2>Title</h2><p>Body&nbsp;text</p></body></html>")
    text = extract(page).pages[0]
    assert "## Title" in text and "Body text" in text and "alert" not in text


def test_plain_text_keeps_fidelity():
    text = extract(FIXTURES / "test.txt").pages[0]
    assert "It has multiple paragraphs." in text


def test_unsupported_and_corrupt(tmp_path):
    with pytest.raises(UnsupportedFormatError):
        extract(tmp_path / "id_rsa")
    broken = tmp_path / "broken.epub"
    broken.write_bytes(b"not a zip")
    with pytest.raises(UnsupportedFormatError):
        extract(broken)


def test_clean_text_removes_pdf_junk():
    raw = "Real sentence here.\n12\nReal sentence here.\nwatermark jane@example.com licensed copy\n​Clean line"
    cleaned = clean_text(raw)
    assert cleaned.splitlines() == ["Real sentence here.", "Clean line"]
