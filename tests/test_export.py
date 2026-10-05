"""Exports to pgvector and to Aegis's player library."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from autorag.export import ExportError, aegis_book_name, aegis_category, export_aegis, export_pgvector
from autorag.kb import KnowledgeBase
from autorag.sql_source import connect_postgres
from tests.helpers import LOREM, HashingEmbedder

AEGIS = "aegis:nomic-embed-text@512"


@pytest.fixture
def library(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "routers.md").write_text(f"# Routers\n\n{LOREM}")
    (docs / "baking.txt").write_text("Sourdough needs a starter, flour, water and salt.")
    return docs


def _query(dsn, sql, params=()):
    conn = connect_postgres(dsn, read_only=False)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [list(r) for r in cur.fetchall()] if cur.description else []
        conn.commit()
        return rows
    finally:
        conn.close()


# --- pgvector ----------------------------------------------------------------------------


def test_pgvector_export_creates_table_index_and_replaces(pg_dsn, tmp_path, library, embedder):
    with KnowledgeBase(tmp_path / "kb.db", embedder=embedder) as kb:
        kb.ingest([library], collection="notes", tags=["lab"])
        report = export_pgvector(kb, "notes", pg_dsn)
        assert (report.documents, report.chunks) == (2, 2)
        assert report.target.endswith("(autorag_chunks)")

        # Nearest neighbour by cosine in Postgres matches the local search
        query_vec = "[" + ",".join(str(float(x)) for x in embedder.embed_query("sourdough starter")) + "]"
        best = _query(
            pg_dsn,
            "SELECT source_name, tags FROM autorag_chunks ORDER BY embedding <=> %s::vector LIMIT 1",
            (query_vec,),
        )
        assert best == [["baking.txt", ["lab"]]]
        indexes = {r[0] for r in _query(pg_dsn, "SELECT indexname FROM pg_indexes WHERE tablename = 'autorag_chunks'")}
        assert "autorag_chunks_embedding_hnsw" in indexes

        kb.remove("baking.txt", collection="notes")
        export_pgvector(kb, "notes", pg_dsn)
        assert _query(pg_dsn, "SELECT count(*) FROM autorag_chunks WHERE collection = 'notes'") == [[1]]


def test_pgvector_refuses_dimension_mismatch(pg_dsn, tmp_path, library):
    with KnowledgeBase(tmp_path / "a.db", embedder=HashingEmbedder(dim=128, name="a")) as kb:
        kb.ingest([library], collection="notes")
        export_pgvector(kb, "notes", pg_dsn)
    with KnowledgeBase(tmp_path / "b.db", embedder=HashingEmbedder(dim=64, name="b")) as kb:
        kb.ingest([library], collection="other")
        with pytest.raises(ExportError, match="vector\\(128\\)"):
            export_pgvector(kb, "other", pg_dsn)
        export_pgvector(kb, "other", pg_dsn, table="autorag_chunks_64")  # a separate table works
    with pytest.raises(ExportError, match="lowercase"):
        export_pgvector(None, "x", pg_dsn, table="bad; drop table x")


# --- Aegis -------------------------------------------------------------------------------


@pytest.fixture
def fake_aegis():
    calls: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else None
            parts = urlsplit(self.path)
            calls.append(
                {
                    "method": self.command,
                    "path": parts.path,
                    "query": parse_qs(parts.query),
                    "body": body,
                    "auth": self.headers.get("Authorization"),
                    "apikey": self.headers.get("apikey"),
                    "prefer": self.headers.get("Prefer"),
                }
            )
            self.send_response(201 if self.command == "POST" else 204)
            self.end_headers()

        do_POST = do_DELETE = _handle

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", calls
    server.shutdown()


def test_aegis_naming_matches_aegis():
    assert aegis_book_name("lost_mine_of_phandelver_v1.2.pdf") == "Lost Mine Of Phandelver"
    assert aegis_book_name("curse-of-strahd.md") == "Curse Of Strahd"
    assert aegis_category("Lost Mine Of Phandelver") == "adventure"
    assert aegis_category("Monster Manual") == "sourcebook"


def test_aegis_export_writes_upload_route_shape(fake_aegis, tmp_path, monkeypatch):
    url, calls = fake_aegis
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "curse-of-strahd.md").write_text("# Barovia\n\nThe mists surround Barovia and its castle.")
    (docs / "homebrew.txt").write_text("My own rules.")
    monkeypatch.setenv("AEGIS_SERVICE_KEY", "test-jwt")
    with KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder(dim=512, name=AEGIS)) as kb:
        kb.ingest([docs], collection="dnd")
        report = export_aegis(kb, "dnd", url=url)

    assert (report.documents, report.chunks) == (1, 1)
    assert report.skipped and "reserved" in report.skipped[0]
    delete, insert, book = calls
    assert delete["method"] == "DELETE" and delete["path"] == "/rest/v1/user_library_chunks"
    assert delete["query"]["source_book"] == ["eq.Curse Of Strahd"]
    assert delete["query"]["user_id"] == ["eq.00000000-0000-4000-8000-000000000001"]
    row = insert["body"][0]
    assert insert["path"] == "/rest/v1/user_library_chunks"
    assert row["source_book"] == "Curse Of Strahd" and row["category"] == "adventure"
    assert row["heading_path"] == ["Barovia"] and row["extraction_status"] == "pending"
    assert len(json.loads(row["embedding"])) == 512
    assert book["path"] == "/rest/v1/user_library_books" and book["query"]["on_conflict"] == ["user_id,name"]
    assert "merge-duplicates" in book["prefer"]
    assert {c["auth"] for c in calls} == {"Bearer test-jwt"} and {c["apikey"] for c in calls} == {"test-jwt"}


def test_aegis_reexport_removes_books_it_exported_earlier(fake_aegis, tmp_path, monkeypatch):
    url, calls = fake_aegis
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "old-rules.md").write_text("# Old\n\nOld rules text.")
    (docs / "new-rules.md").write_text("# New\n\nNew rules text.")
    monkeypatch.setenv("AEGIS_SERVICE_KEY", "k")
    with KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder(dim=512, name=AEGIS)) as kb:
        kb.ingest([docs], collection="rules")
        export_aegis(kb, "rules", url=url)
        kb.remove("old-rules.md", collection="rules")
        calls.clear()
        report = export_aegis(kb, "rules", url=url)
    assert report.removed == ["Old Rules"]
    deletes = [(c["path"], c["query"]) for c in calls if c["method"] == "DELETE"]
    assert ("/rest/v1/user_library_books", {"user_id": ["eq.00000000-0000-4000-8000-000000000001"],
                                             "name": ["eq.Old Rules"]}) in deletes
    # A book AutoRAG never exported (the player's own upload) is never deleted
    assert all("Monster Manual" not in str(q) for _, q in deletes)


def test_aegis_names_follow_javascript_ascii_rules():
    # Aegis's JS regexes have no /u flag, so \b and \w are ASCII-only there
    assert aegis_book_name("über_guide.pdf") == "üBer Guide"


def test_aegis_export_guards(tmp_path, library, monkeypatch):
    monkeypatch.delenv("AEGIS_SERVICE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    with KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder(dim=512, name="fastembed:x")) as kb:
        kb.ingest([library], collection="notes")
        with pytest.raises(ExportError, match="re-index it with --embedder aegis"):
            export_aegis(kb, "notes", key="k")
        with pytest.raises(ExportError, match="No collection"):
            export_aegis(kb, "missing", key="k")
    with KnowledgeBase(tmp_path / "a.db", embedder=HashingEmbedder(dim=512, name=AEGIS)) as kb:
        kb.ingest([library], collection="dnd")
        with pytest.raises(ExportError, match="AEGIS_SERVICE_KEY"):
            export_aegis(kb, "dnd")
        with pytest.raises(ExportError, match="UUID"):
            export_aegis(kb, "dnd", key="k", player_id="player-1")
        with pytest.raises(ExportError, match="not reachable"):
            export_aegis(kb, "dnd", key="k", url="http://127.0.0.1:9")


# Aegis's real table and search function, verbatim from aegis/db/01-init.sql
AEGIS_DDL = """
CREATE TABLE public.user_library_chunks (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    user_id uuid NOT NULL,
    chunk_id text NOT NULL,
    content text NOT NULL,
    token_count integer DEFAULT 0,
    source_book text NOT NULL,
    section_title text,
    heading_path text[],
    embedding public.vector(512),
    created_at timestamp with time zone DEFAULT now(),
    category text DEFAULT 'sourcebook'::text,
    extraction_status text DEFAULT 'pending'::text
);
CREATE FUNCTION public.match_user_library(query_embedding public.vector, p_user_id uuid, match_count integer, filter_books text[] DEFAULT NULL::text[]) RETURNS TABLE(id uuid, content text, source_book text, section_title text, heading_path text[], similarity double precision)
    LANGUAGE plpgsql
    SET search_path TO 'public'
    AS $$
begin
  return query
  select
    ulc.id,
    ulc.content,
    ulc.source_book,
    ulc.section_title,
    ulc.heading_path,
    1 - (ulc.embedding <=> query_embedding) as similarity
  from user_library_chunks ulc
  where
    ulc.embedding is not null
    and ulc.user_id = p_user_id
    and (filter_books is null or ulc.source_book = any(filter_books))
  order by ulc.embedding <=> query_embedding
  limit match_count;
end;
$$;
"""


def test_exported_rows_are_found_by_aegis_match_user_library(pg_dsn, fake_aegis, tmp_path, monkeypatch):
    """The rows the export sends, stored in Aegis's real schema, are what Aegis's own
    search function returns for a matching query."""
    url, calls = fake_aegis
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "curse-of-strahd.md").write_text("# Barovia\n\nThe mists surround Barovia and its castle.")
    (docs / "monster-manual.md").write_text("# Goblins\n\nGoblins are small, cunning raiders.")
    emb = HashingEmbedder(dim=512, name=AEGIS)
    monkeypatch.setenv("AEGIS_SERVICE_KEY", "k")
    with KnowledgeBase(tmp_path / "kb.db", embedder=emb) as kb:
        kb.ingest([docs], collection="dnd")
        export_aegis(kb, "dnd", url=url)

    conn = connect_postgres(pg_dsn, read_only=False)
    cur = conn.cursor()
    for statement in AEGIS_DDL.split(";\nCREATE FUNCTION"):
        cur.execute(statement if statement.startswith("\nCREATE TABLE") else "CREATE FUNCTION" + statement)
    for call in calls:
        if call["method"] == "POST" and call["path"].endswith("user_library_chunks"):
            for r in call["body"]:
                cur.execute(
                    """INSERT INTO user_library_chunks (user_id, chunk_id, content, token_count, source_book,
                       section_title, heading_path, embedding, category, extraction_status)
                       VALUES (%s, %s, %s, %s, %s, %s, %s::text[], %s::vector, %s, %s)""",
                    (r["user_id"], r["chunk_id"], r["content"], r["token_count"], r["source_book"],
                     r["section_title"], r["heading_path"], r["embedding"], r["category"], r["extraction_status"]),
                )
    query = "[" + ",".join(str(float(x)) for x in emb.embed_query("goblin raiders")) + "]"
    cur.execute(
        "SELECT source_book, similarity FROM match_user_library(%s::vector, %s::uuid, 2)",
        (query, "00000000-0000-4000-8000-000000000001"),
    )
    results = cur.fetchall()
    conn.commit()
    conn.close()
    assert results[0][0] == "Monster Manual" and results[0][1] > results[1][1]


def test_pgvector_large_dims_skip_hnsw_and_batches_span_boundaries(pg_dsn, tmp_path):
    from autorag.kb import TextItem

    items = [TextItem(f"sql:bulk/{i}", f"row {i}", f"Row number {i} talks about topic {i % 7}.") for i in range(1203)]
    with KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder(dim=2048, name="big")) as kb:
        kb.ingest_texts(items, collection="bulk")
        report = export_pgvector(kb, "bulk", pg_dsn, table="bulk_chunks")
    assert report.chunks == 1203  # three INSERT batches of up to 500
    assert any("HNSW" in note for note in report.skipped)
    assert _query(pg_dsn, "SELECT count(*) FROM bulk_chunks") == [[1203]]
    indexes = {r[0] for r in _query(pg_dsn, "SELECT indexname FROM pg_indexes WHERE tablename = 'bulk_chunks'")}
    assert "bulk_chunks_embedding_hnsw" not in indexes
