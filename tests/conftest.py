"""Shared fixtures."""

import pytest

from tests.helpers import HashingEmbedder


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def pg_dsn():
    """A fresh, empty database with pgvector, dropped afterwards. Skips without
    AUTORAG_TEST_PG_DSN."""
    import uuid
    from urllib.parse import urlsplit, urlunsplit

    from tests.helpers import pg_admin_dsn

    admin = pg_admin_dsn()
    if not admin:
        pytest.skip("AUTORAG_TEST_PG_DSN not set")
    pg8000 = pytest.importorskip("pg8000.dbapi")
    from autorag.sql_source import connect_postgres

    name = f"autorag_test_{uuid.uuid4().hex[:10]}"
    conn = connect_postgres(admin, read_only=False)
    conn.autocommit = True
    conn.cursor().execute(f"CREATE DATABASE {name}")
    conn.close()
    dsn = urlunsplit(urlsplit(admin)._replace(path=f"/{name}"))
    db = connect_postgres(dsn, read_only=False)
    db.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
    db.commit()
    db.close()
    yield dsn
    conn = connect_postgres(admin, read_only=False)
    conn.autocommit = True
    conn.cursor().execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    conn.close()
    assert pg8000
