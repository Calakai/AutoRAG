"""Structure-aware chunking with no model download.

Replaces docling-core's HybridChunker, which fetched a HuggingFace tokenizer on
first use — a hard failure offline or behind a proxy, and a `transformers`
install for a token count. Token counts here are a conservative estimate (the
larger of chars/4 and words×1.3), which tracks BERT-style wordpiece tokenizers
closely enough to size chunks for the embedding models AutoRAG uses.

Strategies:
- ``hybrid`` (default): pack paragraphs up to ``max_tokens``, keep sections apart,
  but merge a section too small to stand alone into its neighbour.
- ``hierarchical``: one chunk per section, split only when oversized.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STRATEGIES = ("hybrid", "hierarchical")

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")


def _outside_fences(lines: list[str]):
    """Yield (line, in_fence) — a `# comment` inside a code fence is not a heading."""
    in_fence = False
    for line in lines:
        if _FENCE.match(line.strip()):
            yield line, True
            in_fence = not in_fence
            continue
        yield line, in_fence
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, len(text) // 4, int(len(text.split()) * 1.3))


def looks_like_heading(line: str) -> bool:
    """AutoRAG's original heading heuristic for text with no markup."""
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


def _heuristic_level(line: str) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)", line)
    return min(match.group(1).count(".") + 1, 4) if match else 1


@dataclass
class Block:
    kind: str  # "heading" | "paragraph"
    text: str
    page: int | None  # 1-based; None for page-less formats
    level: int = 0
    heading_path: tuple[str, ...] = ()


@dataclass
class Chunk:
    text: str
    token_count: int
    heading_path: list[str]
    page_start: int | None
    page_end: int | None
    element_types: list[str] = field(default_factory=lambda: ["paragraph"])


def parse_blocks(pages: list[str], paged: bool) -> list[Block]:
    """Split page texts into heading and paragraph blocks.

    Markdown ``#`` headings win when the document has any; otherwise headings are
    guessed. Wrapped lines from paged sources are re-joined (with de-hyphenation);
    page-less text keeps its line breaks.
    """
    use_markdown = any(
        not fenced and _MD_HEADING.match(line.strip())
        for page in pages
        for line, fenced in _outside_fences(page.split("\n"))
    )
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []

    for page_index, text in enumerate(pages):
        page_no = page_index + 1 if paged else None
        para: list[str] = []

        def flush_para() -> None:
            if not para:
                return
            joined = _join_lines(para) if paged else "\n".join(para)
            blocks.append(
                Block("paragraph", joined, page_no, heading_path=tuple(t for _, t in stack))
            )
            para.clear()

        for raw, fenced in _outside_fences(text.split("\n")):
            line = raw.strip()
            if fenced:  # code stays verbatim, blank lines included, in one paragraph
                para.append(raw.rstrip())
                continue
            if not line:
                flush_para()
                continue
            md = _MD_HEADING.match(line) if use_markdown else None
            if md or (not use_markdown and looks_like_heading(line)):
                flush_para()
                level = len(md.group(1)) if md else _heuristic_level(line)
                title = md.group(2) if md else line
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                blocks.append(Block("heading", title, page_no, level=level))
                continue
            para.append(line if paged else raw.rstrip())
        flush_para()
    return blocks


def _join_lines(lines: list[str]) -> str:
    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-") and line[:1].islower():
            out = out[:-1] + line
        else:
            out += " " + line
    return out


def blocks_to_markdown(blocks: list[Block]) -> str:
    parts = [
        ("#" * block.level + " " + block.text) if block.kind == "heading" else block.text
        for block in blocks
    ]
    return "\n\n".join(parts) + ("\n" if parts else "")


@dataclass
class _Piece:
    text: str
    tokens: int
    page: int | None
    path: tuple[str, ...]


def _split_long(text: str, max_tokens: int) -> list[str]:
    """Split oversized text on sentence boundaries, falling back to words."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if estimate_tokens(candidate) <= max_tokens:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if estimate_tokens(sentence) <= max_tokens:
            current = sentence
            continue
        words, current = sentence.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and estimate_tokens(candidate) > max_tokens:
                pieces.append(current)
                current = word
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, tokens: int) -> str:
    """The last ~`tokens` worth of words of `text`, for chunk overlap."""
    if tokens <= 0:
        return ""
    words = text.split()
    take = max(1, int(tokens / 1.3))
    return " ".join(words[-take:])


def _label(path: tuple[str, ...], common: tuple[str, ...]) -> str:
    return " › ".join(path[len(common) :])


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return tuple(out)


def chunk_blocks(
    blocks: list[Block],
    max_tokens: int = 512,
    overlap_tokens: int = 64,
    strategy: str = "hybrid",
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}; expected one of {STRATEGIES}")
    if max_tokens < 16:
        raise ValueError("max_tokens must be at least 16")
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))
    merge_below = max_tokens // 4

    pieces: list[_Piece] = []
    for block in blocks:
        if block.kind != "paragraph":
            continue
        for text in _split_long(block.text, max_tokens):
            pieces.append(_Piece(text, estimate_tokens(text), block.page, block.heading_path))

    chunks: list[Chunk] = []
    current: list[_Piece] = []
    path: tuple[str, ...] = ()

    def flush() -> None:
        if not current:
            return
        text = "\n\n".join(p.text for p in current)
        pages = [p.page for p in current if p.page is not None]
        chunks.append(
            Chunk(
                text=text,
                token_count=estimate_tokens(text),
                heading_path=list(path),
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
            )
        )

    section: tuple[str, ...] = ()  # heading path of the last piece added
    for piece in pieces:
        tokens = sum(p.tokens for p in current)
        if current and piece.path != section:
            # A merged chunk keeps only the shared heading path, so each section's
            # own headings are written into the text instead of being lost.
            common = _common_prefix(path, piece.path)
            head, own = _label(path, common), _label(piece.path, common)
            extra = estimate_tokens(head) + estimate_tokens(own)
            if strategy == "hybrid" and tokens < merge_below and tokens + piece.tokens + extra <= max_tokens:
                if head:
                    current.insert(0, _Piece(head, estimate_tokens(head), current[0].page, common))
                if own:
                    current.append(_Piece(own, estimate_tokens(own), piece.page, common))
                path = common
            else:
                flush()
                current = []
        elif current and tokens + piece.tokens > max_tokens:
            last = current[-1]
            carry = _tail(last.text, overlap_tokens)
            flush()
            current = []
            path = piece.path
            if carry and estimate_tokens(carry) + piece.tokens <= max_tokens:
                current.append(_Piece(carry, estimate_tokens(carry), last.page, path))
        if not current:
            path = piece.path
        current.append(piece)
        section = piece.path
    flush()
    return chunks
