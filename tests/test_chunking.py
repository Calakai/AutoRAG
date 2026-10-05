"""Structure-aware chunker: headings, sizing, overlap, page provenance."""

import pytest

from autorag.chunking import (
    blocks_to_markdown,
    chunk_blocks,
    estimate_tokens,
    parse_blocks,
)
from tests.helpers import LOREM


def test_markdown_headings_build_heading_paths():
    text = "# Guide\n\nIntro text.\n\n## Install\n\nRun the installer.\n\n## Use\n\nOpen the app."
    blocks = parse_blocks([text], paged=False)
    paths = [b.heading_path for b in blocks if b.kind == "paragraph"]
    assert paths == [("Guide",), ("Guide", "Install"), ("Guide", "Use")]


def test_markdown_mode_ignores_title_case_lines():
    # A Title Case sentence in a markdown doc is not a heading
    blocks = parse_blocks(["# Doc\n\nHello World\nmore text"], paged=False)
    assert [b.kind for b in blocks] == ["heading", "paragraph"]


def test_heuristic_headings_for_unmarked_text():
    blocks = parse_blocks(["CHAPTER ONE\nsome body text here.\n2.1 Details\nmore body text."], paged=True)
    headings = [(b.text, b.level) for b in blocks if b.kind == "heading"]
    assert headings == [("CHAPTER ONE", 1), ("2.1 Details", 2)]


def test_paged_lines_rejoin_with_dehyphenation():
    blocks = parse_blocks(["the rout-\ning table is\nconsulted."], paged=True)
    assert blocks[0].text == "the routing table is consulted."


def test_chunks_respect_max_tokens_and_overlap():
    body = " ".join([LOREM] * 12)
    blocks = parse_blocks([f"# Networking\n\n{body}"], paged=False)
    chunks = chunk_blocks(blocks, max_tokens=120, overlap_tokens=20)
    assert len(chunks) > 3
    assert all(c.token_count <= 120 for c in chunks)
    assert all(c.heading_path == ["Networking"] for c in chunks)
    # Each chunk after the first starts with the tail of the previous one
    tail = chunks[0].text.split()[-5:]
    assert chunks[1].text.split()[:15] and " ".join(tail) in chunks[1].text


def test_hybrid_merges_tiny_sections_hierarchical_does_not():
    text = "# A\n\nShort.\n\n# B\n\nAlso short.\n\n# C\n\n" + LOREM
    blocks = parse_blocks([text], paged=False)
    hybrid = chunk_blocks(blocks, max_tokens=200, strategy="hybrid")
    hierarchical = chunk_blocks(blocks, max_tokens=200, strategy="hierarchical")
    assert len(hierarchical) == 3
    assert len(hybrid) < len(hierarchical)
    # Merged chunks keep only the heading path they share, and carry each
    # section's heading in the text exactly once
    assert hybrid[0].heading_path == []
    assert hybrid[0].text.startswith("A\n\nShort.\n\nB\n\nAlso short.")


def test_merge_labels_each_section_once():
    text = "# A\n\nOne.\n\nTwo.\n\n# B\n\nThree.\n\nFour."
    chunk = chunk_blocks(parse_blocks([text], paged=False), max_tokens=200)[0]
    assert chunk.text == "A\n\nOne.\n\nTwo.\n\nB\n\nThree.\n\nFour."


def test_page_provenance_spans_pages():
    pages = ["Intro\n" + LOREM, LOREM, LOREM]
    blocks = parse_blocks(pages, paged=True)
    chunks = chunk_blocks(blocks, max_tokens=2000)
    assert chunks[0].page_start == 1 and chunks[0].page_end == 3


def test_unpaged_chunks_have_no_page_numbers():
    chunks = chunk_blocks(parse_blocks(["just text"], paged=False))
    assert chunks[0].page_start is None and chunks[0].page_end is None


def test_oversized_word_run_is_still_split():
    blocks = parse_blocks(["word " * 2000], paged=False)
    chunks = chunk_blocks(blocks, max_tokens=100, overlap_tokens=0)
    assert all(c.token_count <= 100 for c in chunks)


def test_markdown_export_and_validation():
    blocks = parse_blocks(["# T\n\nBody."], paged=False)
    assert blocks_to_markdown(blocks) == "# T\n\nBody.\n"
    with pytest.raises(ValueError):
        chunk_blocks(blocks, strategy="semantic")
    assert estimate_tokens("") == 0 and estimate_tokens("a b c d") >= 4


def test_code_fence_comments_are_not_headings():
    text = "# Install\n\n```bash\n# install deps\npip install x\n\n# more\n```\n\nAfter."
    blocks = parse_blocks([text], paged=False)
    assert [b.text for b in blocks if b.kind == "heading"] == ["Install"]
    assert "# install deps" in blocks[1].text and "# more" in blocks[1].text
    assert blocks[2].heading_path == ("Install",)


def test_overlap_applies_to_short_paragraphs():
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(12))
    chunks = chunk_blocks(parse_blocks([paragraphs], paged=False), max_tokens=120, overlap_tokens=30)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text.split()[0] in prev.text.split()[-40:]
