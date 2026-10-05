"""KnowledgeBase: ingest documents, embed them locally, and search them.

The one API the CLI, the MCP server and Python callers share::

    from autorag.kb import KnowledgeBase

    with KnowledgeBase() as kb:                 # ~/.autorag/knowledge.db, fastembed
        kb.ingest(["~/Documents/manuals"], collection="manuals")
        for hit in kb.search("reset the router", collection="manuals"):
            print(hit.citation(), hit.text[:80])
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from autorag.config import ProcessingConfig
from autorag.embed import Embedder, EmbeddingUnavailableError, resolve_embedder
from autorag.extract import SUPPORTED_EXTENSIONS
from autorag.pipeline import compute_sha256, process_document
from autorag.store import (
    ChunkRecord,
    CollectionInfo,
    DocumentInfo,
    EmbedderMismatchError,
    SearchHit,
    SQLiteStore,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "default"


def data_home() -> Path:
    return Path(os.environ.get("AUTORAG_HOME") or Path.home() / ".autorag").expanduser()


def default_db_path() -> Path:
    return data_home() / "knowledge.db"


def collect_files(paths: Iterable[str | Path], recursive: bool = False) -> list[Path]:
    """Expand files and directories into supported document files.

    Hidden files and directories are skipped when walking a directory, so pointing
    at a home folder never sweeps up dotfiles. Explicitly named files are taken
    as given (and rejected later if unsupported).
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            walker = path.rglob("*") if recursive else path.glob("*")
            for candidate in sorted(walker):
                rel = candidate.relative_to(path)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(candidate)
        elif path.is_file():
            found.append(path)
        else:
            logger.warning("Skipping %s (not found)", raw)
    # De-duplicate while keeping order
    return list(dict.fromkeys(p.resolve() for p in found))


@dataclass
class IngestOutcome:
    path: str
    status: str  # added | updated | retagged | unchanged | failed
    chunks: int = 0
    pages: int = 0
    ocr_pages_skipped: int = 0
    error: str = ""


def document_key(collection: str, source_path: str, stem: str) -> str:
    """A readable id prefix unique per (collection, path): two README.md files in
    different folders must never share chunk ids."""
    digest = hashlib.sha1(f"{collection}\0{source_path}".encode()).hexdigest()[:8]
    return f"{stem}-{digest}"


def embedding_text(text: str, heading_path: list[str], source_name: str) -> str:
    """What gets embedded: the chunk with its document and section context, so a
    chunk that never repeats its heading still matches questions about it."""
    context = " › ".join([source_name, *heading_path])
    return f"{context}\n\n{text}"


class KnowledgeBase:
    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: Embedder | str | None = None,
        config: ProcessingConfig | None = None,
    ) -> None:
        self.store = SQLiteStore(Path(db_path).expanduser() if db_path else default_db_path())
        self.embedder = embedder if embedder is not None and not isinstance(embedder, str) else resolve_embedder(embedder)
        self.config = config or ProcessingConfig()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.store.close()

    # --- ingest ----------------------------------------------------------------------

    def ingest(
        self,
        paths: Iterable[str | Path],
        collection: str = DEFAULT_COLLECTION,
        tags: Iterable[str] = (),
        recursive: bool = False,
        force: bool = False,
        on_file: Callable[[int, int, Path], None] | None = None,
    ) -> list[IngestOutcome]:
        """Add or refresh documents. Unchanged files (same content hash) are skipped
        unless `force`; changed files replace their previous chunks atomically."""
        files = collect_files(paths, recursive=recursive)
        existing = self.store.get_collection(collection)
        if existing and existing.embedder != self.embedder.name:
            # Fail before any extraction, OCR or embedding is spent on the batch
            raise EmbedderMismatchError(
                f"Collection '{collection}' was built with {existing.embedder}; "
                f"the active embedder is {self.embedder.name}."
            )
        outcomes: list[IngestOutcome] = []
        for index, path in enumerate(files):
            if on_file:
                on_file(index, len(files), path)
            outcomes.append(self._ingest_one(path, collection, list(tags), force))
        return outcomes

    def _ingest_one(self, path: Path, collection: str, tags: list[str], force: bool) -> IngestOutcome:
        source_path = str(path)
        try:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported format: {path.suffix or '(none)'}")
            source_hash = compute_sha256(path)
            existing = self.store.get_document(collection, source_path)
            if existing and existing.source_hash == source_hash and not force:
                if tags and sorted(existing.tags) != sorted(tags):
                    self.store.update_tags(existing.id, tags)  # no need to re-embed
                    return IngestOutcome(source_path, "retagged", existing.total_chunks, existing.total_pages)
                return IngestOutcome(source_path, "unchanged", existing.total_chunks, existing.total_pages)

            result = process_document(path, self.config, source_hash=source_hash)
            key = document_key(collection, source_path, path.stem)
            records = [
                ChunkRecord(
                    chunk_id=f"{key}_chunk_{c.metadata['chunk_index']:03d}",
                    text=c.text,
                    token_count=c.token_count,
                    page_start=c.metadata["page_start"],
                    page_end=c.metadata["page_end"],
                    section_title=c.metadata["section_title"],
                    heading_path=c.metadata["heading_path"],
                )
                for c in result.chunks
            ]
            vectors = self.embedder.embed_documents(
                [embedding_text(r.text, r.heading_path, path.name) for r in records]
            )
            dim = int(vectors.shape[1]) if len(records) else self._probe_dim()
            self.store.ensure_collection(collection, self.embedder.name, dim)
            self.store.replace_document(
                collection,
                source_path,
                path.name,
                source_hash,
                records,
                vectors.reshape(len(records), dim) if len(records) else vectors.reshape(0, dim),
                total_pages=result.total_pages,
                ocr_used=result.ocr_used,
                ocr_pages_skipped=result.ocr_pages_skipped,
                tags=tags or (existing.tags if existing else []),
            )
            return IngestOutcome(
                source_path,
                "updated" if existing else "added",
                len(records),
                result.total_pages,
                result.ocr_pages_skipped,
            )
        except (EmbeddingUnavailableError, EmbedderMismatchError):
            raise  # affects every file — stop instead of failing them one by one
        except Exception as exc:
            logger.warning("Failed to ingest %s: %s", path, exc)
            return IngestOutcome(source_path, "failed", error=str(exc))

    def _probe_dim(self) -> int:
        return int(self.embedder.embed_query("dimension probe").shape[-1])

    # --- query -----------------------------------------------------------------------

    def searchable_collections(self, collection: str | None = None) -> list[str]:
        """Collections the current embedder can search. A named collection built
        with another embedder raises EmbedderMismatchError."""
        if collection:
            info = self.store.get_collection(collection)
            if info is None:
                raise KeyError(f"No collection named {collection!r}")
            if info.embedder != self.embedder.name:
                raise EmbedderMismatchError(
                    f"Collection '{collection}' was built with {info.embedder}; "
                    f"the active embedder is {self.embedder.name}."
                )
            return [collection]
        all_collections = self.store.collections()
        usable = [c.name for c in all_collections if c.embedder == self.embedder.name]
        if all_collections and not usable:
            # Saying "no results" here would hide a library that exists
            built_with = sorted({c.embedder for c in all_collections})
            raise EmbedderMismatchError(
                f"No collection was built with the active embedder {self.embedder.name}; "
                f"existing collections use {', '.join(built_with)}. Pass --embedder to match."
            )
        return usable

    def search(
        self,
        query: str,
        collection: str | None = None,
        k: int = 5,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        collections = self.searchable_collections(collection)
        if not collections or not query.strip():
            return []
        vector = self.embedder.embed_query(query) if mode != "keyword" else None
        return self.store.search(query, vector, collections, k=k, mode=mode)

    def read(self, chunk_id: str, neighbors: int = 1, collection: str | None = None) -> list[SearchHit]:
        return self.store.read_chunk(chunk_id, neighbors=neighbors, collection=collection)

    def documents(self, collection: str | None = None) -> list[DocumentInfo]:
        return self.store.documents(collection)

    def collections(self) -> list[CollectionInfo]:
        return self.store.collections()

    def remove(self, source: str, collection: str = DEFAULT_COLLECTION) -> int:
        looks_like_path = "/" in source or os.sep in source or Path(source).expanduser().exists()
        resolved = str(Path(source).expanduser().resolve()) if looks_like_path else source
        return self.store.delete_document(collection, resolved)
