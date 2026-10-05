"""SQL tables as a knowledge source: each row becomes one text document.

Supports SQLite files (stdlib) and Postgres (the optional ``[postgres]`` extra,
pg8000 — BSD-licensed; psycopg is LGPL and ruled out by the no-copyleft rule).
Connections are read-only: SQLite opens with ``mode=ro`` and Postgres sessions
are set to ``READ ONLY``, so a query can never modify the source database.

A row's identity is ``sql:<label>/<id>``. Connection strings — which may carry a
password — are never stored or printed; ``redact_dsn`` is used wherever one has
to be shown.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from autorag.kb import TextItem


class SQLSourceError(ValueError):
    """The SQL source is misconfigured or unreachable."""


def redact_dsn(dsn: str) -> str:
    """The DSN with any password replaced, safe to log."""
    parts = urlsplit(dsn)
    if parts.password is None:
        return dsn
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return urlunsplit(parts._replace(netloc=netloc))


def is_postgres(dsn: str) -> bool:
    return dsn.startswith(("postgres://", "postgresql://"))


def _sqlite_path(dsn: str) -> Path:
    if re.search(r"\b(host|user|password|dbname)\s*=", dsn):
        # A libpq keyword DSN: refuse without echoing it (it may hold a password)
        raise SQLSourceError("Keyword-style DSNs aren't supported; use postgresql://user:pass@host/db")
    if dsn.startswith("sqlite:///"):
        return Path(dsn[len("sqlite:///") :]).expanduser()
    if "://" in dsn:
        raise SQLSourceError(f"Unsupported DSN scheme in {redact_dsn(dsn)}; use postgres://… or a SQLite file path")
    return Path(dsn).expanduser()


def _ssl_context(sslmode: str | None):
    """libpq sslmode semantics: require = encrypt without verifying, verify-ca =
    verify the chain, verify-full = chain + hostname. Unset/disable/allow/prefer
    connect in plaintext (pg8000 cannot fall back the way libpq's prefer does)."""
    import ssl

    if sslmode in (None, "", "disable", "allow", "prefer"):
        return None
    if sslmode not in ("require", "verify-ca", "verify-full"):
        raise SQLSourceError(f"Unsupported sslmode {sslmode!r}")
    ctx = ssl.create_default_context()
    if sslmode != "verify-full":
        ctx.check_hostname = False
    if sslmode == "require":
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _pg_params(dsn: str) -> dict:
    if not is_postgres(dsn):
        raise SQLSourceError("Postgres connections must be postgres:// or postgresql:// URLs")
    parts = urlsplit(dsn)
    query = dict(p.split("=", 1) for p in parts.query.split("&") if "=" in p)
    return {
        "user": unquote(parts.username or "postgres"),
        "password": unquote(parts.password) if parts.password else None,
        "host": parts.hostname or "localhost",
        "port": parts.port or 5432,
        "database": unquote(parts.path.lstrip("/")) or None,
        "ssl_context": _ssl_context(query.get("sslmode")),
    }


def connect_postgres(dsn: str, read_only: bool = True):
    """A pg8000 DB-API connection from a postgres:// URL (used by exports)."""
    try:
        import pg8000.dbapi
    except ImportError as exc:
        raise SQLSourceError("Postgres needs the [postgres] extra: pip install 'autorag[postgres]'") from exc
    try:
        conn = pg8000.dbapi.connect(**_pg_params(dsn))
    except SQLSourceError:
        raise
    except Exception as exc:
        raise SQLSourceError(f"Could not connect to {redact_dsn(dsn)}: {exc}") from exc
    if read_only:
        cur = conn.cursor()
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        conn.commit()
    return conn


def _fetch_postgres(dsn: str, query: str) -> list[dict]:
    """Read-only by construction: the session is READ ONLY, and the query runs as a
    prepared statement, which Postgres limits to ONE statement — so a query string
    cannot `COMMIT; BEGIN READ WRITE; …` its way out (verified against Postgres 16)."""
    try:
        import pg8000.native
    except ImportError as exc:
        raise SQLSourceError("Postgres needs the [postgres] extra: pip install 'autorag[postgres]'") from exc
    try:
        conn = pg8000.native.Connection(**_pg_params(dsn))
    except SQLSourceError:
        raise
    except Exception as exc:
        raise SQLSourceError(f"Could not connect to {redact_dsn(dsn)}: {exc}") from exc
    try:
        conn.run("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        statement = conn.prepare(query)
        rows = statement.run()
        names = [c["name"] for c in statement.columns or []]
        return [dict(zip(names, row)) for row in rows or []]
    except Exception as exc:
        message = exc.args[0].get("M", str(exc)) if exc.args and isinstance(exc.args[0], dict) else str(exc)
        raise SQLSourceError(f"Query failed: {message}") from exc
    finally:
        conn.close()


def fetch_rows(dsn: str, query: str) -> list[dict]:
    """Run a read-only query and return rows as dicts."""
    if is_postgres(dsn):
        return _fetch_postgres(dsn, query)
    path = _sqlite_path(dsn)
    if not path.is_file():
        raise SQLSourceError(f"SQLite database not found: {path}")
    # as_uri() percent-encodes '#', '?' and '%' so they stay part of the path
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute(query).fetchall()]
        except sqlite3.Error as exc:
            raise SQLSourceError(f"Query failed: {exc}") from exc


_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass
class RowMapping:
    """How rows become documents."""

    label: str  # stable name for this source, part of every row's identity
    id_column: str
    text_columns: list[str]
    title_column: str | None = None
    meta_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _LABEL.match(self.label):
            raise SQLSourceError("--label must be 1-64 characters of letters, digits, '.', '_' or '-'")
        if not self.text_columns:
            raise SQLSourceError("At least one --text-columns column is required")

    @property
    def prefix(self) -> str:
        return f"sql:{self.label}/"


def row_to_item(row: dict, mapping: RowMapping) -> TextItem | None:
    """Render one row as a markdown document; None when it has no text."""
    needed = [mapping.id_column, *mapping.text_columns, *mapping.meta_columns]
    if mapping.title_column:
        needed.append(mapping.title_column)
    missing = [c for c in needed if c not in row]
    if missing:
        raise SQLSourceError(f"Query result has no column(s): {', '.join(missing)} (got {', '.join(row)})")

    row_id = row[mapping.id_column]
    if row_id is None:
        raise SQLSourceError(f"Row with NULL {mapping.id_column}; the id column must identify every row")
    title = str(row[mapping.title_column]).strip() if mapping.title_column and row[mapping.title_column] else ""

    parts: list[str] = [f"# {title}"] if title else []
    meta = [f"{c}: {row[c]}" for c in mapping.meta_columns if row[c] not in (None, "")]
    if meta:
        parts.append("\n".join(meta))
    texts = [(c, str(row[c]).strip()) for c in mapping.text_columns if row[c] not in (None, "")]
    texts = [(c, t) for c, t in texts if t]
    if not texts:
        return None
    if len(mapping.text_columns) == 1:
        parts.append(texts[0][1])
    else:
        parts.extend(f"## {c}\n\n{t}" for c, t in texts)

    name = title or f"{mapping.label} #{row_id}"
    return TextItem(source_path=f"{mapping.prefix}{row_id}", source_name=name, text="\n\n".join(parts))


def rows_to_items(rows: list[dict], mapping: RowMapping) -> list[TextItem]:
    items, seen = [], set()
    for row in rows:
        item = row_to_item(row, mapping)
        if item is None:
            continue
        if item.source_path in seen:
            raise SQLSourceError(f"Duplicate id {item.source_path!r}: --id-column must be unique per row")
        seen.add(item.source_path)
        items.append(item)
    return items
