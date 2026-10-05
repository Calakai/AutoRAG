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
