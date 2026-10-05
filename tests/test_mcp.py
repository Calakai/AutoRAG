"""MCP tool behaviour and server registration."""

import asyncio
import os
import re

import pytest

from autorag import mcp_server
from autorag.kb import KnowledgeBase
from tests.helpers import LOREM


@pytest.fixture
def kb(tmp_path, embedder):
    with KnowledgeBase(db_path=tmp_path / "kb.db", embedder=embedder) as kb:
        yield kb


@pytest.fixture
def doc(tmp_path):
    path = tmp_path / "docs" / "routers.md"
    path.parent.mkdir()
    path.write_text(f"# Routers\n\n{LOREM}")
    return path


def test_search_before_ingest_explains_empty_library(kb):
    assert "Ingest documents first" in mcp_server.search_knowledge(kb, "routers")


def test_ingest_search_read_remove_roundtrip(kb, doc):
    report = mcp_server.ingest_documents(kb, [str(doc.parent)], collection="net", tags=["lab"])
    assert "added: routers.md" in report

    found = mcp_server.search_knowledge(kb, "longest prefix routing", collection="net")
    assert "[1] routers.md › Routers" in found and "<excerpt>" in found
    chunk_id = re.search(r"chunk_id: (\S+)", found).group(1)
    assert chunk_id.startswith("routers-") and chunk_id.endswith("_chunk_001")

    window = mcp_server.read_chunk(kb, chunk_id)
    assert "longest prefix wins" in window

    assert "net: 1 documents" in mcp_server.list_collections(kb)
    assert "tags: lab" in mcp_server.list_documents(kb, "net")
    assert mcp_server.remove_document(kb, "routers.md", "net") == "Removed 1 document(s)."
    assert "No document" in mcp_server.remove_document(kb, "routers.md", "net")


def test_bad_inputs_return_messages_not_exceptions(kb):
    assert mcp_server.search_knowledge(kb, "x", mode="fuzzy").startswith("mode must be")
    assert mcp_server.search_knowledge(kb, "x", collection="nope").startswith("Error:")
    assert "No chunk" in mcp_server.read_chunk(kb, "missing")


def test_allowed_roots_restrict_ingest(kb, doc, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("AUTORAG_ALLOWED_ROOTS", str(allowed))
    assert mcp_server.ingest_documents(kb, [str(doc)]).startswith("Refused")
    inside = allowed / "ok.md"
    inside.write_text("# Ok\n\nAllowed content here.")
    assert "added: ok.md" in mcp_server.ingest_documents(kb, [str(inside)])
    monkeypatch.setenv("AUTORAG_ALLOWED_ROOTS", os.pathsep.join([str(allowed), str(doc.parent)]))
    assert "added" in mcp_server.ingest_documents(kb, [str(doc)])


def test_symlink_cannot_escape_allowed_roots(kb, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("# Secret\n\nprivate api token")
    (allowed / "notes.md").symlink_to(secret)
    monkeypatch.setenv("AUTORAG_ALLOWED_ROOTS", str(allowed))
    assert mcp_server.ingest_documents(kb, [str(allowed)]).startswith("Refused")
    assert kb.documents() == []


def test_server_registers_named_tools(kb, doc):
    pytest.importorskip("mcp")
    server = mcp_server.build_server(kb)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "search_knowledge",
        "read_chunk",
        "list_collections",
        "list_documents",
        "ingest_documents",
        "remove_document",
    }
    mcp_server.ingest_documents(kb, [str(doc)])
    result = asyncio.run(server.call_tool("search_knowledge", {"query": "routing tables"}))
    # SDK 2.x returns a result object; 1.x returns (content, structured)
    content = result.content if hasattr(result, "content") else result[0]
    text = content[0].text
    assert "routers.md" in text
