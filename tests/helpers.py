"""Test helpers: an offline embedder and builders for real document files."""

from __future__ import annotations

import re
import zipfile
import zlib
from pathlib import Path

import numpy as np

from autorag.embed import normalize


class HashingEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words score higher.

    Lets search tests run with no model download (CI and sandboxes can't reach
    HuggingFace)."""

    def __init__(self, dim: int = 128, name: str = "test:hashing") -> None:
        self.dim = dim
        self.name = name
        self.document_calls: list[list[str]] = []

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        return vec

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return normalize(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vector(text))[0]


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal, valid multi-page PDF with a real text layer."""
    objects: list[bytes] = []
    n_pages = len(pages)
    font_id = 3
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        lines = text.split("\n")
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for line in lines:
            ops.append(f"({_pdf_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def make_epub(path: Path, chapters: list[tuple[str, str]]) -> Path:
    """Write an EPUB whose spine order differs from archive order."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
        )
        manifest = "".join(
            f'<item id="c{i}" href="text/ch{i}.xhtml" media-type="application/xhtml+xml"/>'
            for i in range(len(chapters))
        )
        spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
        zf.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
            f"<manifest>{manifest}</manifest><spine>{spine}</spine></package>",
        )
        # Archive order reversed, so only the spine gives the right order
        for i in reversed(range(len(chapters))):
            title, body = chapters[i]
            zf.writestr(
                f"OEBPS/text/ch{i}.xhtml",
                f"<html><head><title>x</title><style>p{{}}</style></head><body>"
                f"<h1>{title}</h1><p>{body}</p></body></html>",
            )
    return path


def make_fb2(path: Path, body: str) -> Path:
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        f"<body><section><title><p>Chapter One</p></title><p>{body}</p></section></body>"
        "</FictionBook>",
        encoding="utf-8",
    )
    return path


def make_xps(path: Path, pages: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for i, text in enumerate(pages, start=1):
            zf.writestr(
                f"Documents/1/Pages/{i}.fpage",
                '<FixedPage xmlns="http://schemas.microsoft.com/xps/2005/06" Width="816" Height="1056">'
                f'<Glyphs UnicodeString="{text}" OriginX="10" OriginY="20"/></FixedPage>',
            )
    return path


LOREM = (
    "Routers forward packets between networks using routing tables. "
    "Each table entry maps a destination prefix to a next hop and an interface. "
    "When several entries match, the longest prefix wins and the packet follows it. "
)
