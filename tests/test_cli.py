"""CLI commands end to end, with the offline embedder swapped in."""

import json

import pytest

from autorag import cli, kb as kb_module
from tests.helpers import LOREM, HashingEmbedder


@pytest.fixture(autouse=True)
def offline_embedder(monkeypatch):
    monkeypatch.setattr(kb_module, "resolve_embedder", lambda spec=None: HashingEmbedder())


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "routers.md").write_text(f"# Routers\n\n{LOREM}")
    (d / "notes.rst").write_text("Release notes for the build tooling.")
    (d / "skip.xyz").write_text("ignored")
    return d


def test_index_search_docs_rm(tmp_path, docs, capsys):
    db = str(tmp_path / "kb.db")
    cli.main(["index", str(docs), "-c", "net", "--db", db, "--json", "-q"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert {l["status"] for l in lines} == {"added"} and len(lines) == 2  # .rst now picked up

    cli.main(["search", "routing tables", "--db", db, "--json"])
    hit = json.loads(capsys.readouterr().out.splitlines()[0])
    assert hit["source_name"] == "routers.md" and hit["collection"] == "net"

    cli.main(["collections", "--db", db])
    assert "net: 2 documents" in capsys.readouterr().out
    cli.main(["docs", "-c", "net", "--db", db])
    assert "routers.md" in capsys.readouterr().out

    cli.main(["rm", "routers.md", "-c", "net", "--db", db])
    with pytest.raises(SystemExit):
        cli.main(["rm", "routers.md", "-c", "net", "--db", db])


def test_errors_exit_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["search", "x", "-c", "missing", "--db", str(tmp_path / "kb.db")])
    assert exc.value.code == 2
    assert "No collection named 'missing'" in capsys.readouterr().err


def test_process_writes_chunk_folders(tmp_path, docs, capsys):
    out = tmp_path / "out"
    cli.main(["process", str(docs), "-o", str(out), "-q", "--json"])
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(results) == 2
    assert (out / "routers" / "chunks_combined.jsonl").exists()


def test_info_without_pymupdf(tmp_path, capsys):
    from tests.helpers import make_pdf

    pdf = make_pdf(tmp_path / "a.pdf", ["one", "two", "three"])
    cli.main(["info", str(pdf)])
    info = json.loads(capsys.readouterr().out)
    assert info["pages"] == 3 and info["supported"]


def test_process_keeps_verbose_flag(tmp_path, docs):
    args = cli.build_parser().parse_args(["process", str(docs), "-v"])
    assert args.verbose is True


def test_index_sql_from_env_dsn_and_sync(tmp_path, monkeypatch, capsys):
    import sqlite3

    src = tmp_path / "crm.db"
    conn = sqlite3.connect(src)
    conn.executescript(
        "CREATE TABLE notes (id INTEGER, title TEXT, body TEXT);"
        "INSERT INTO notes VALUES (1, 'Call Dana', 'Dana wants the router quote by Friday.');"
        "INSERT INTO notes VALUES (2, 'Invoice', 'Invoice 42 is overdue.');"
    )
    conn.commit()
    monkeypatch.setenv("CRM_DSN", str(src))
    db = str(tmp_path / "kb.db")
    args = ["index-sql", "--dsn", "env:CRM_DSN", "--query", "SELECT * FROM notes", "--label", "crm",
            "--id-column", "id", "--text-columns", "body", "--title-column", "title",
            "-c", "crm", "--sync", "--json", "-q", "--db", db]
    cli.main(args)
    assert {json.loads(l)["status"] for l in capsys.readouterr().out.splitlines()} == {"added"}
    conn.execute("DELETE FROM notes WHERE id = 2")
    conn.commit()
    cli.main(args)
    statuses = {json.loads(l)["path"]: json.loads(l)["status"] for l in capsys.readouterr().out.splitlines()}
    assert statuses == {"sql:crm/1": "unchanged", "sql:crm/2": "removed"}


def test_export_errors_exit_cleanly(tmp_path, docs, monkeypatch, capsys):
    monkeypatch.delenv("AUTORAG_EXPORT_DSN", raising=False)
    db = str(tmp_path / "kb.db")
    cli.main(["index", str(docs), "-c", "net", "--db", db, "-q"])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(["export", "-c", "net", "--to", "aegis", "--db", db])
    assert exc.value.code == 2 and "re-index it with --embedder aegis" in capsys.readouterr().err
    # An unset env:VAR must fail, never fall back to $AUTORAG_EXPORT_DSN
    monkeypatch.setenv("AUTORAG_EXPORT_DSN", "postgresql://prod.example/db")
    with pytest.raises(SystemExit):
        cli.main(["export", "-c", "net", "--to", "env:NOPE", "--db", db])
    assert "NOPE is not set" in capsys.readouterr().err
    for bad in ("Aegis", "./backup.db"):
        with pytest.raises(SystemExit):
            cli.main(["export", "-c", "net", "--to", bad, "--db", db])
        assert "--to must be" in capsys.readouterr().err
