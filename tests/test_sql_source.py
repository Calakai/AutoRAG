"""SQL rows as documents: mapping, read-only access, incremental sync."""

import sqlite3

import pytest

from autorag.kb import KnowledgeBase
from autorag.sql_source import (
    RowMapping,
    SQLSourceError,
    connect_postgres,
    fetch_rows,
    redact_dsn,
    row_to_item,
    rows_to_items,
)


@pytest.fixture
def notes_db(tmp_path):
    path = tmp_path / "notes.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT, body TEXT, status TEXT);
        INSERT INTO notes VALUES (1, 'Router reset', 'Hold the reset button for ten seconds.', 'done');
        INSERT INTO notes VALUES (2, 'Sourdough', 'Feed the starter flour and water daily.', 'open');
        INSERT INTO notes VALUES (3, 'Empty', NULL, 'open');
        """
    )
    conn.commit()
    conn.close()
    return path


MAPPING = RowMapping(label="notes", id_column="id", text_columns=["body"], title_column="title", meta_columns=("status",))


def test_row_rendering_and_identity():
    item = row_to_item({"id": 7, "title": "Reset", "body": "Hold it.", "status": "done"}, MAPPING)
    assert item.source_path == "sql:notes/7" and item.source_name == "Reset"
    assert item.text == "# Reset\n\nstatus: done\n\nHold it."
    multi = RowMapping(label="t", id_column="id", text_columns=["a", "b"])
    assert row_to_item({"id": 1, "a": "x", "b": "y"}, multi).text == "## a\n\nx\n\n## b\n\ny"
    assert row_to_item({"id": 1, "a": None, "b": " "}, multi) is None  # nothing to index


def test_mapping_errors_are_clear():
    with pytest.raises(SQLSourceError, match="no column"):
        row_to_item({"id": 1, "title": "x"}, MAPPING)
    with pytest.raises(SQLSourceError, match="NULL id"):
        row_to_item({"id": None, "title": "x", "body": "y", "status": ""}, MAPPING)
    with pytest.raises(SQLSourceError, match="Duplicate id"):
        rows_to_items([{"id": 1, "title": "a", "body": "b", "status": ""}] * 2, MAPPING)
    with pytest.raises(SQLSourceError):
        RowMapping(label="bad label!", id_column="id", text_columns=["x"])


def test_sqlite_is_opened_read_only(notes_db):
    rows = fetch_rows(str(notes_db), "SELECT * FROM notes ORDER BY id")
    assert [r["id"] for r in rows] == [1, 2, 3]
    assert fetch_rows(f"sqlite:///{notes_db}", "SELECT count(*) AS n FROM notes")[0]["n"] == 3
    with pytest.raises(SQLSourceError, match="readonly"):
        fetch_rows(str(notes_db), "DELETE FROM notes")
    with pytest.raises(SQLSourceError, match="not found"):
        fetch_rows(str(notes_db.parent / "missing.db"), "SELECT 1")


def test_postgres_query_cannot_escape_read_only(pg_dsn):
    conn = connect_postgres(pg_dsn, read_only=False)
    cur = conn.cursor()
    cur.execute("CREATE TABLE notes (id int, body text)")
    cur.execute("INSERT INTO notes VALUES (1, 'keep me')")
    conn.commit()
    conn.close()
    attacks = [
        "COMMIT; BEGIN READ WRITE; DELETE FROM notes; COMMIT; SELECT 1 AS id",
        "WITH d AS (DELETE FROM notes RETURNING *) SELECT * FROM d",
        "DELETE FROM notes",
    ]
    for query in attacks:
        with pytest.raises(SQLSourceError):
            fetch_rows(pg_dsn, query)
    assert fetch_rows(pg_dsn, "SELECT count(*) AS n FROM notes") == [{"n": 1}]


def test_unsafe_dsns_are_refused_without_echoing_secrets(tmp_path):
    with pytest.raises(SQLSourceError) as exc:
        fetch_rows("host=db user=app password=s3cret dbname=prod", "SELECT 1")
    assert "s3cret" not in str(exc.value)
    weird = tmp_path / "notes#2.db"
    sqlite3.connect(weird).executescript("CREATE TABLE t (x); INSERT INTO t VALUES (1);")
    assert fetch_rows(str(weird), "SELECT x FROM t") == [{"x": 1}]


def test_sslmode_follows_libpq():
    import ssl

    from autorag.sql_source import _ssl_context

    assert _ssl_context(None) is None and _ssl_context("prefer") is None
    require = _ssl_context("require")
    assert require.verify_mode == ssl.CERT_NONE and not require.check_hostname
    assert _ssl_context("verify-ca").verify_mode == ssl.CERT_REQUIRED and not _ssl_context("verify-ca").check_hostname
    assert _ssl_context("verify-full").check_hostname
    with pytest.raises(SQLSourceError):
        _ssl_context("sometimes")


def test_redact_dsn():
    assert redact_dsn("postgresql://cal:s3cr%40t@db:5432/x") == "postgresql://cal:***@db:5432/x"
    assert redact_dsn("postgresql://db/x") == "postgresql://db/x"


def test_ingest_rows_incrementally_with_sync(tmp_path, notes_db, embedder):
    with KnowledgeBase(tmp_path / "kb.db", embedder=embedder) as kb:
        items = rows_to_items(fetch_rows(str(notes_db), "SELECT * FROM notes"), MAPPING)
        first = kb.ingest_texts(items, collection="notes", sync_prefix=MAPPING.prefix)
        assert sorted(o.status for o in first) == ["added", "added"]  # empty row skipped
        hit = kb.search("reset button", collection="notes", k=1)[0]
        assert hit.source_name == "Router reset" and hit.source_path == "sql:notes/1"

        conn = sqlite3.connect(notes_db)
        conn.execute("UPDATE notes SET body = 'Feed it rye flour.' WHERE id = 2")
        conn.execute("DELETE FROM notes WHERE id = 1")
        conn.commit()
        conn.close()
        items = rows_to_items(fetch_rows(str(notes_db), "SELECT * FROM notes"), MAPPING)
        second = {o.path: o.status for o in kb.ingest_texts(items, collection="notes", sync_prefix=MAPPING.prefix)}
        assert second == {"sql:notes/2": "updated", "sql:notes/1": "removed"}
        assert [d.source_path for d in kb.documents("notes")] == ["sql:notes/2"]
        third = kb.ingest_texts(items, collection="notes", sync_prefix=MAPPING.prefix)
        assert [o.status for o in third] == ["unchanged"]


def test_sync_never_touches_other_sources(tmp_path, embedder):
    from autorag.kb import TextItem

    with KnowledgeBase(tmp_path / "kb.db", embedder=embedder) as kb:
        kb.ingest_texts([TextItem("sql:a/1", "a1", "alpha text")], collection="c")
        kb.ingest_texts([TextItem("sql:b/1", "b1", "bravo text")], collection="c", sync_prefix="sql:b/")
        kb.ingest_texts([], collection="c", sync_prefix="sql:b/")
        assert [d.source_path for d in kb.documents("c")] == ["sql:a/1"]
        with pytest.raises(ValueError, match="outside the sync prefix"):
            kb.ingest_texts([TextItem("sql:a/2", "a2", "x")], collection="c", sync_prefix="sql:b/")


def test_postgres_source_is_read_only(pg_dsn, tmp_path, embedder):
    conn = connect_postgres(pg_dsn, read_only=False)
    cur = conn.cursor()
    cur.execute("CREATE TABLE tickets (id int PRIMARY KEY, subject text, body text)")
    cur.execute("INSERT INTO tickets VALUES (1, 'VPN down', 'Restart the VPN gateway service.')")
    conn.commit()
    conn.close()

    rows = fetch_rows(pg_dsn, "SELECT * FROM tickets")
    assert rows == [{"id": 1, "subject": "VPN down", "body": "Restart the VPN gateway service."}]
    with pytest.raises(SQLSourceError, match="read-only"):
        fetch_rows(pg_dsn, "DELETE FROM tickets")
    mapping = RowMapping(label="tickets", id_column="id", text_columns=["body"], title_column="subject")
    with KnowledgeBase(tmp_path / "kb.db", embedder=embedder) as kb:
        kb.ingest_texts(rows_to_items(rows, mapping), collection="it")
        assert kb.search("vpn gateway", k=1)[0].source_name == "VPN down"
