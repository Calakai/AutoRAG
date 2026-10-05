"""SQLite knowledge store: documents, chunks, vectors and keyword search in one file.

Vectors are float32 BLOBs searched by brute-force dot product in numpy, the same
recipe MonkChat uses — no native vector extension to install or sign, and fast
enough for personal libraries (tens of thousands of chunks). Keyword search is
FTS5 bm25. Hybrid search fuses the two rankings by reciprocal-rank fusion.

Each collection records the embedder that built it; searching or adding to it
with a different embedder is refused rather than silently comparing vectors from
different models.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEARCH_MODES = ("hybrid", "vector", "keyword")
RRF_K = 60  # standard reciprocal-rank-fusion constant

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    name        TEXT PRIMARY KEY,
    embedder    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id                 INTEGER PRIMARY KEY,
    collection         TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
    source_path        TEXT NOT NULL,
    source_name        TEXT NOT NULL,
    source_hash        TEXT NOT NULL,
    total_pages        INTEGER NOT NULL DEFAULT 0,
    total_chunks       INTEGER NOT NULL DEFAULT 0,
    ocr_used           INTEGER NOT NULL DEFAULT 0,
    ocr_pages_skipped  INTEGER NOT NULL DEFAULT 0,
    tags               TEXT NOT NULL DEFAULT '[]',
    added_at           TEXT NOT NULL,
    UNIQUE (collection, source_path)
);
CREATE TABLE IF NOT EXISTS chunks (
    id             INTEGER PRIMARY KEY,
    document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,
    chunk_id       TEXT NOT NULL,
    text           TEXT NOT NULL,
    token_count    INTEGER NOT NULL,
    page_start     INTEGER,
    page_end       INTEGER,
    section_title  TEXT NOT NULL DEFAULT '',
    heading_path   TEXT NOT NULL DEFAULT '[]',
    embedding      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id, seq);
CREATE TABLE IF NOT EXISTS exports (
    target      TEXT NOT NULL,
    collection  TEXT NOT NULL,
    item        TEXT NOT NULL,
    PRIMARY KEY (target, collection, item)
);
CREATE INDEX IF NOT EXISTS chunks_chunk_id ON chunks(chunk_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, section_title, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, section_title) VALUES (new.id, new.text, new.section_title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, section_title)
    VALUES ('delete', old.id, old.text, old.section_title);
END;
"""


class EmbedderMismatchError(ValueError):
    """A collection was built with a different embedder than the one in use."""


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    section_title: str
    heading_path: list[str]


@dataclass
class DocumentInfo:
    id: int
    collection: str
    source_path: str
    source_name: str
    source_hash: str
    total_pages: int
    total_chunks: int
    ocr_used: bool
    ocr_pages_skipped: int
    tags: list[str]
    added_at: str


@dataclass
class CollectionInfo:
    name: str
    embedder: str
    dim: int
    created_at: str
    documents: int = 0
    chunks: int = 0


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    collection: str
    source_name: str
    source_path: str
    page_start: int | None
    page_end: int | None
    section_title: str
    heading_path: list[str]
    score: float
    vector_score: float | None = None
    keyword_rank: int | None = None
    tags: list[str] = field(default_factory=list)

    def citation(self) -> str:
        where = self.source_name
        if self.page_start:
            where += f", p.{self.page_start}" + (
                f"–{self.page_end}" if self.page_end and self.page_end != self.page_start else ""
            )
        if self.heading_path:
            where += " › " + " › ".join(self.heading_path)
        return where


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fts_query(text: str) -> str:
    """User text → a safe FTS5 query: quoted terms OR'd together (bm25 rewards
    chunks matching more of them). Never passes FTS syntax through."""
    terms = [t for t in re.findall(r"\w+", text.lower()) if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._vector_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if str(path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.executescript(_FTS_SCHEMA)
                self.has_fts = True
            except sqlite3.OperationalError:  # SQLite built without FTS5
                self.has_fts = False

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- collections -----------------------------------------------------------------

    def get_collection(self, name: str) -> CollectionInfo | None:
        row = self._conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
        return CollectionInfo(row["name"], row["embedder"], row["dim"], row["created_at"]) if row else None

    def ensure_collection(self, name: str, embedder: str, dim: int) -> CollectionInfo:
        with self._lock:
            existing = self.get_collection(name)
            if existing is None:
                self._conn.execute(
                    "INSERT INTO collections(name, embedder, dim, created_at) VALUES (?, ?, ?, ?)",
                    (name, embedder, dim, _now()),
                )
                self._conn.commit()
                return CollectionInfo(name, embedder, dim, _now())
            if existing.embedder != embedder or existing.dim != dim:
                raise EmbedderMismatchError(
                    f"Collection '{name}' was built with {existing.embedder} ({existing.dim} dims), "
                    f"not {embedder} ({dim} dims). Use the same embedder or a different collection."
                )
            return existing

    def collections(self) -> list[CollectionInfo]:
        rows = self._conn.execute(
            """
            SELECT c.*, COUNT(DISTINCT d.id) AS documents, COALESCE(SUM(d.total_chunks), 0) AS chunks
            FROM collections c LEFT JOIN documents d ON d.collection = c.name
            GROUP BY c.name ORDER BY c.name
            """
        ).fetchall()
        return [
            CollectionInfo(r["name"], r["embedder"], r["dim"], r["created_at"], r["documents"], r["chunks"])
            for r in rows
        ]

    def delete_collection(self, name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            self._conn.commit()
            self._vector_cache.clear()
            return cur.rowcount > 0

    # --- documents -------------------------------------------------------------------

    def _doc(self, row: sqlite3.Row) -> DocumentInfo:
        return DocumentInfo(
            id=row["id"],
            collection=row["collection"],
            source_path=row["source_path"],
            source_name=row["source_name"],
            source_hash=row["source_hash"],
            total_pages=row["total_pages"],
            total_chunks=row["total_chunks"],
            ocr_used=bool(row["ocr_used"]),
            ocr_pages_skipped=row["ocr_pages_skipped"],
            tags=json.loads(row["tags"]),
            added_at=row["added_at"],
        )

    def get_document(self, collection: str, source_path: str) -> DocumentInfo | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE collection = ? AND source_path = ?", (collection, source_path)
        ).fetchone()
        return self._doc(row) if row else None

    def documents(self, collection: str | None = None) -> list[DocumentInfo]:
        if collection:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE collection = ? ORDER BY source_name", (collection,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM documents ORDER BY collection, source_name").fetchall()
        return [self._doc(r) for r in rows]

    def replace_document(
        self,
        collection: str,
        source_path: str,
        source_name: str,
        source_hash: str,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
        total_pages: int = 0,
        ocr_used: bool = False,
        ocr_pages_skipped: int = 0,
        tags: list[str] | None = None,
    ) -> DocumentInfo:
        """Insert or atomically replace one document and all of its chunks."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings differ in length")
        coll = self.get_collection(collection)
        if coll is None:
            raise KeyError(f"Unknown collection {collection!r}")
        if len(chunks) and embeddings.shape[1] != coll.dim:
            raise EmbedderMismatchError(f"Embeddings have {embeddings.shape[1]} dims; collection expects {coll.dim}")
        vectors = np.asarray(embeddings, dtype=np.float32)
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM documents WHERE collection = ? AND source_path = ?", (collection, source_path)
            )
            cur = self._conn.execute(
                """
                INSERT INTO documents(collection, source_path, source_name, source_hash, total_pages,
                                      total_chunks, ocr_used, ocr_pages_skipped, tags, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection,
                    source_path,
                    source_name,
                    source_hash,
                    total_pages,
                    len(chunks),
                    int(ocr_used),
                    ocr_pages_skipped,
                    json.dumps(list(tags or [])),
                    _now(),
                ),
            )
            doc_id = cur.lastrowid
            self._conn.executemany(
                """
                INSERT INTO chunks(document_id, seq, chunk_id, text, token_count, page_start, page_end,
                                   section_title, heading_path, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        doc_id,
                        seq,
                        c.chunk_id,
                        c.text,
                        c.token_count,
                        c.page_start,
                        c.page_end,
                        c.section_title,
                        json.dumps(c.heading_path),
                        vectors[seq].tobytes(),
                    )
                    for seq, c in enumerate(chunks)
                ],
            )
            self._vector_cache.clear()
        return self.get_document(collection, source_path)  # type: ignore[return-value]

    def update_tags(self, document_id: int, tags: list[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE documents SET tags = ? WHERE id = ?", (json.dumps(list(tags)), document_id))

    def delete_document(self, collection: str, source: str) -> int:
        """Delete by source path or bare file name; returns documents removed."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM documents WHERE collection = ? AND (source_path = ? OR source_name = ?)",
                (collection, source, source),
            )
            self._vector_cache.clear()
            return cur.rowcount

    # --- reading ---------------------------------------------------------------------

    _HIT_COLUMNS = """
        c.id, c.chunk_id, c.text, c.page_start, c.page_end, c.section_title, c.heading_path,
        d.collection, d.source_name, d.source_path, d.tags
    """

    def _hit(self, row: sqlite3.Row, score: float, vector_score=None, keyword_rank=None) -> SearchHit:
        return SearchHit(
            chunk_id=row["chunk_id"],
            text=row["text"],
            collection=row["collection"],
            source_name=row["source_name"],
            source_path=row["source_path"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_title=row["section_title"],
            heading_path=json.loads(row["heading_path"]),
            score=score,
            vector_score=vector_score,
            keyword_rank=keyword_rank,
            tags=json.loads(row["tags"]),
        )

    def read_chunk(self, chunk_id: str, neighbors: int = 0, collection: str | None = None) -> list[SearchHit]:
        """A chunk plus up to `neighbors` chunks either side from the same document."""
        sql = "SELECT c.document_id, c.seq FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.chunk_id = ?"
        params: list = [chunk_id]
        if collection:
            sql += " AND d.collection = ?"
            params.append(collection)
        row = self._conn.execute(sql + " LIMIT 1", params).fetchone()
        if row is None:
            return []
        rows = self._conn.execute(
            f"""
            SELECT {self._HIT_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = ? AND c.seq BETWEEN ? AND ? ORDER BY c.seq
            """,
            (row["document_id"], row["seq"] - neighbors, row["seq"] + neighbors),
        ).fetchall()
        return [self._hit(r, score=0.0) for r in rows]

    def exported_items(self, target: str, collection: str | None = None) -> set[str]:
        """Names AutoRAG wrote to an export target (all collections when None)."""
        if collection is None:
            rows = self._conn.execute("SELECT item FROM exports WHERE target = ?", (target,))
        else:
            rows = self._conn.execute(
                "SELECT item FROM exports WHERE target = ? AND collection = ?", (target, collection)
            )
        return {r["item"] for r in rows}

    def set_exported_items(self, target: str, collection: str, items: set[str]) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM exports WHERE target = ? AND collection = ?", (target, collection))
            self._conn.executemany(
                "INSERT INTO exports(target, collection, item) VALUES (?, ?, ?)",
                [(target, collection, item) for item in sorted(items)],
            )

    def iter_chunks(self, collection: str):
        """Every chunk of a collection with its document and vector, in document order
        — the export path."""
        cur = self._conn.execute(
            """
            SELECT c.chunk_id, c.text, c.token_count, c.page_start, c.page_end, c.section_title,
                   c.heading_path, c.embedding, d.source_name, d.source_path, d.tags
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.collection = ? ORDER BY d.source_path, c.seq
            """,
            (collection,),
        )
        for row in cur:
            yield {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "token_count": row["token_count"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "section_title": row["section_title"],
                "heading_path": json.loads(row["heading_path"]),
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
                "source_name": row["source_name"],
                "source_path": row["source_path"],
                "tags": json.loads(row["tags"]),
            }

    # --- search ----------------------------------------------------------------------

    def _vectors(self, collections: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        cached = self._vector_cache.get(collections)
        if cached is not None:
            return cached
        marks = ",".join("?" * len(collections))
        rows = self._conn.execute(
            f"""
            SELECT c.id, c.embedding FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE d.collection IN ({marks})
            """,
            collections,
        ).fetchall()
        if rows:
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        else:
            ids, matrix = np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        self._vector_cache[collections] = (ids, matrix)
        return ids, matrix

    def _rows_by_id(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT {self._HIT_COLUMNS} FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id IN ({marks})",
            ids,
        ).fetchall()
        return {r["id"]: r for r in rows}

    def vector_search(self, query_vector: np.ndarray, collections: list[str], limit: int) -> list[tuple[int, float]]:
        with self._lock:
            ids, matrix = self._vectors(tuple(sorted(collections)))
        if not len(ids):
            return []
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        top = np.argsort(-scores)[:limit]
        return [(int(ids[i]), float(scores[i])) for i in top]

    def keyword_search(self, query: str, collections: list[str], limit: int) -> list[int]:
        match = fts_query(query)
        if not self.has_fts or not match or not collections:
            return []
        marks = ",".join("?" * len(collections))
        rows = self._conn.execute(
            f"""
            SELECT chunks_fts.rowid AS id FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ? AND d.collection IN ({marks})
            ORDER BY bm25(chunks_fts) LIMIT ?
            """,
            (match, *collections, limit),
        ).fetchall()
        return [r["id"] for r in rows]

    def search(
        self,
        query: str,
        query_vector: np.ndarray | None,
        collections: list[str],
        k: int = 5,
        mode: str = "hybrid",
    ) -> list[SearchHit]:
        if mode not in SEARCH_MODES:
            raise ValueError(f"Unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
        if not collections:
            return []
        pool = max(k * 4, 20)
        vector_hits = (
            self.vector_search(query_vector, collections, pool)
            if mode != "keyword" and query_vector is not None
            else []
        )
        keyword_hits = self.keyword_search(query, collections, pool) if mode != "vector" else []

        vector_scores = dict(vector_hits)
        keyword_ranks = {cid: rank for rank, cid in enumerate(keyword_hits, start=1)}
        fused: dict[int, float] = {}
        for rank, (cid, _) in enumerate(vector_hits, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        for cid, rank in keyword_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        if mode == "vector":
            ordered = [cid for cid, _ in vector_hits[:k]]
        elif mode == "keyword":
            ordered = keyword_hits[:k]
        else:
            ordered = sorted(fused, key=lambda cid: -fused[cid])[:k]
        rows = self._rows_by_id(ordered)
        return [
            self._hit(
                rows[cid],
                score=round(vector_scores[cid] if mode == "vector" else fused[cid], 6),
                vector_score=round(vector_scores[cid], 4) if cid in vector_scores else None,
                keyword_rank=keyword_ranks.get(cid),
            )
            for cid in ordered
            if cid in rows
        ]
