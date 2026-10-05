"""Push a collection out of the local library into another database.

The SQLite library stays the source of truth; an export replaces that
collection's rows in the target, so re-running it is how the target is refreshed.

- ``export_pgvector`` — any Postgres with pgvector (local, Docker, Supabase).
  AutoRAG creates and owns the table (default ``autorag_chunks``) and an HNSW
  cosine index. Needs the ``[postgres]`` extra.
- ``export_aegis`` — Aegis's player library (``user_library_chunks`` and
  ``user_library_books``), written through Aegis's own PostgREST endpoint the way
  its upload route writes them. Stdlib HTTP only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from autorag.extract import SUPPORTED_EXTENSIONS
from autorag.kb import KnowledgeBase
from autorag.sql_source import connect_postgres, redact_dsn

AEGIS_EMBEDDER = "aegis:nomic-embed-text@512"
AEGIS_DEFAULT_URL = "http://localhost:3001"
AEGIS_DEFAULT_PLAYER = "00000000-0000-4000-8000-000000000001"  # Aegis's LOCAL_PLAYER_ID sentinel
# Same reserved-name rule and category hints as aegis/src/app/api/uploads/route.ts
_AEGIS_RESERVED = re.compile(r"^(homebrew|srd(\s*5(\.[12])?(\.1)?)?)$", re.I)
_AEGIS_ADVENTURE_HINTS = ("adventure", "module", "campaign", "curse", "descent", "tomb", "lost mine", "dragon of")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
HNSW_MAX_DIM = 2000  # pgvector's limit for an HNSW index on `vector`
INSERT_BATCH = 500
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class ExportError(ValueError):
    """The export target is misconfigured, unreachable or incompatible."""


@dataclass
class ExportReport:
    target: str
    collection: str
    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def _collection_or_fail(kb: KnowledgeBase, collection: str):
    info = kb.store.get_collection(collection)
    if info is None:
        raise ExportError(f"No collection named {collection!r}")
    return info


def _vector_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


# --- pgvector ----------------------------------------------------------------------------


def export_pgvector(kb: KnowledgeBase, collection: str, dsn: str, table: str = "autorag_chunks") -> ExportReport:
    if not _IDENT.match(table):
        raise ExportError("Table name must be lowercase letters, digits and underscores")
    info = _collection_or_fail(kb, collection)
    rows = list(kb.store.iter_chunks(collection))
    report = ExportReport(target=f"{redact_dsn(dsn)} ({table})", collection=collection)

    conn = connect_postgres(dsn, read_only=False)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if not cur.fetchall():
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as exc:
                raise ExportError(f"pgvector is not installed in this database and could not be created: {exc}") from exc
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id            bigserial PRIMARY KEY,
                collection    text NOT NULL,
                chunk_id      text NOT NULL,
                source_name   text NOT NULL,
                source_path   text NOT NULL,
                content       text NOT NULL,
                token_count   integer NOT NULL DEFAULT 0,
                page_start    integer,
                page_end      integer,
                section_title text NOT NULL DEFAULT '',
                heading_path  text[] NOT NULL DEFAULT '{{}}',
                tags          text[] NOT NULL DEFAULT '{{}}',
                embedder      text NOT NULL,
                embedding     vector({info.dim}) NOT NULL,
                exported_at   timestamptz NOT NULL DEFAULT now(),
                UNIQUE (collection, chunk_id)
            )
            """
        )
        cur.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a
            WHERE a.attrelid = %s::regclass AND a.attname = 'embedding'
            """,
            (table,),
        )
        column_type = cur.fetchall()[0][0]
        if column_type != f"vector({info.dim})":
            raise ExportError(
                f"{table}.embedding is {column_type}, but collection '{collection}' has {info.dim}-dim vectors "
                f"({info.embedder}). Export to a different --table."
            )
        if info.dim <= HNSW_MAX_DIM:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)"
            )
        else:
            report.skipped.append(
                f"HNSW index: pgvector indexes at most {HNSW_MAX_DIM} dims; {info.dim}-dim vectors are searched exactly"
            )
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_collection ON {table} (collection)")

        cur.execute(f"DELETE FROM {table} WHERE collection = %s", (collection,))
        columns = (
            "collection, chunk_id, source_name, source_path, content, token_count, page_start, page_end, "
            "section_title, heading_path, tags, embedder, embedding"
        )
        placeholder = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::text[], %s::text[], %s, %s::vector)"
        for i in range(0, len(rows), INSERT_BATCH):  # multi-row VALUES: one round trip per batch
            batch = rows[i : i + INSERT_BATCH]
            params = [
                value
                for r in batch
                for value in (
                    collection,
                    r["chunk_id"],
                    r["source_name"],
                    r["source_path"],
                    r["text"],
                    r["token_count"],
                    r["page_start"],
                    r["page_end"],
                    r["section_title"],
                    r["heading_path"],
                    r["tags"],
                    info.embedder,
                    _vector_literal(r["embedding"]),
                )
            ]
            cur.execute(f"INSERT INTO {table} ({columns}) VALUES " + ", ".join([placeholder] * len(batch)), params)
        conn.commit()  # one transaction: the target never holds a half-exported collection
    except ExportError:
        conn.rollback()
        raise
    except Exception as exc:
        conn.rollback()
        raise ExportError(f"Export to {redact_dsn(dsn)} failed: {exc}") from exc
    finally:
        conn.close()

    report.documents = len({r["source_path"] for r in rows})
    report.chunks = len(rows)
    return report


# --- Aegis -------------------------------------------------------------------------------


def aegis_book_name(source_name: str) -> str:
    """Port of Aegis's deriveTitleFromName (src/lib/library/extract.ts).

    Aegis only strips .pdf/.txt/.epub because those are all it accepts; AutoRAG
    reads more formats, so any supported extension is dropped first."""
    suffix = os.path.splitext(source_name)[1]
    name = source_name[: -len(suffix)] if suffix.lower() in SUPPORTED_EXTENSIONS else source_name
    # re.ASCII: JavaScript regexes without /u treat \b, \w and \d as ASCII-only,
    # and the names must come out exactly as Aegis derives them.
    name = re.sub(r"\.(pdf|txt|epub)$", "", name, flags=re.I)
    name = re.sub(r"[_-]+", " ", name)
    name = re.sub(r"\bv?\d+(?:\.\d+)+\b", "", name, flags=re.I | re.ASCII)
    name = re.sub(r"\s+", " ", name).strip()
    return re.sub(r"\b\w", lambda m: m.group(0).upper(), name, flags=re.ASCII) or "Untitled"


def aegis_category(book_name: str) -> str:
    lower = book_name.lower()
    return "adventure" if any(h in lower for h in _AEGIS_ADVENTURE_HINTS) else "sourcebook"


class _PostgREST:
    def __init__(self, url: str, key: str, timeout: float = 60.0) -> None:
        self.base = url.rstrip("/") + "/rest/v1"  # Aegis's nginx hop strips this prefix
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def request(self, method: str, path: str, body=None, prefer: str = "return=minimal") -> None:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base}/{path}", data=data, method=method, headers={**self.headers, "Prefer": prefer}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise ExportError(f"Aegis rejected {method} {path.split('?')[0]} ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ExportError(f"Aegis is not reachable at {self.base}: {exc}") from exc


def export_aegis(
    kb: KnowledgeBase,
    collection: str,
    url: str | None = None,
    key: str | None = None,
    player_id: str | None = None,
    batch_size: int = 200,
) -> ExportReport:
    """Write each document of the collection as an Aegis library book."""
    info = _collection_or_fail(kb, collection)
    if info.embedder != AEGIS_EMBEDDER:
        raise ExportError(
            f"Collection '{collection}' was built with {info.embedder}. Aegis searches {AEGIS_EMBEDDER} vectors: "
            "re-index it with --embedder aegis into a new collection first."
        )
    url = url or os.environ.get("AEGIS_URL") or AEGIS_DEFAULT_URL
    key = key or os.environ.get("AEGIS_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise ExportError("Set AEGIS_SERVICE_KEY to the service_role JWT from Aegis's `npm run db:jwt`")
    player_id = player_id or os.environ.get("AEGIS_PLAYER_ID") or AEGIS_DEFAULT_PLAYER
    if not _UUID.match(player_id):
        raise ExportError(f"AEGIS_PLAYER_ID must be a UUID, got {player_id!r}")

    books: OrderedDict[str, list[dict]] = OrderedDict()
    for row in kb.store.iter_chunks(collection):
        books.setdefault(row["source_path"], []).append(row)

    api = _PostgREST(url, key)
    report = ExportReport(target=f"Aegis at {url}", collection=collection)
    names: dict[str, str] = {}
    for source_path, rows in books.items():
        book = aegis_book_name(rows[0]["source_name"])
        if _AEGIS_RESERVED.match(book):
            report.skipped.append(f"{rows[0]['source_name']}: '{book}' is a reserved name in Aegis")
            continue
        if book in names and names[book] != source_path:
            report.skipped.append(f"{rows[0]['source_name']}: another document already exports as '{book}'")
            continue
        names[book] = source_path
        category = aegis_category(book)

        # Same replace semantics as the upload route: clear the book, then insert.
        query = urllib.parse.urlencode({"user_id": f"eq.{player_id}", "source_book": f"eq.{book}"})
        api.request("DELETE", f"user_library_chunks?{query}")
        for i in range(0, len(rows), batch_size):
            api.request(
                "POST",
                "user_library_chunks",
                [
                    {
                        "user_id": player_id,
                        "chunk_id": r["chunk_id"],
                        "content": r["text"],
                        "token_count": r["token_count"],
                        "source_book": book,
                        "section_title": r["section_title"] or None,
                        "heading_path": r["heading_path"],
                        "embedding": _vector_literal(r["embedding"]),
                        "category": category,
                        "extraction_status": "pending",
                    }
                    for r in rows[i : i + batch_size]
                ],
            )
        api.request(
            "POST",
            "user_library_books?on_conflict=user_id,name",
            {
                "user_id": player_id,
                "name": book,
                "title": book,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            prefer="resolution=merge-duplicates,return=minimal",
        )
        report.documents += 1
        report.chunks += len(rows)

    # Books this collection exported before but no longer contains (a removed file, a
    # deleted SQL row, a changed title) come out of Aegis too. Only names AutoRAG
    # itself recorded are touched — never a book the player uploaded in Aegis.
    target = f"aegis:{url.rstrip('/')}#{player_id}"
    previous = kb.store.exported_items(target, collection)
    others = kb.store.exported_items(target) - previous
    for book in sorted(previous - set(names) - others):
        query = urllib.parse.urlencode({"user_id": f"eq.{player_id}", "source_book": f"eq.{book}"})
        api.request("DELETE", f"user_library_chunks?{query}")
        query = urllib.parse.urlencode({"user_id": f"eq.{player_id}", "name": f"eq.{book}"})
        api.request("DELETE", f"user_library_books?{query}")
        report.removed.append(book)
    kb.store.set_exported_items(target, collection, set(names))
    return report
