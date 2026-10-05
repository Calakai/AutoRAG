"""KnowledgeBase + SQLite store: ingest, refresh, search, provenance, isolation."""

import sqlite3

import pytest

from autorag.kb import KnowledgeBase, collect_files, embedding_text
from autorag.store import EmbedderMismatchError, fts_query
from tests.helpers import LOREM, HashingEmbedder, make_pdf


@pytest.fixture
def library(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "routers.md").write_text(f"# Routers\n\n{LOREM}\n\n## Tables\n\nThe routing table lists prefixes.")
    (docs / "baking.txt").write_text(
        "Sourdough bread needs a starter, flour, water and salt. Bake the loaf in a hot oven."
    )
    make_pdf(docs / "firewall.pdf", ["Firewall Rules\nA firewall filters packets by port and address. " * 3])
    (docs / ".secret.md").write_text("hidden token")
    (docs / "nested").mkdir()
    (docs / "nested" / "deep.md").write_text("# Deep\n\nNested note about vlan tagging.")
    return docs


@pytest.fixture
def kb(tmp_path, embedder):
    with KnowledgeBase(db_path=tmp_path / "kb.db", embedder=embedder) as kb:
        yield kb


def test_collect_files_skips_hidden_and_respects_recursion(library):
    flat = {p.name for p in collect_files([library])}
    assert flat == {"routers.md", "baking.txt", "firewall.pdf"}
    deep = {p.name for p in collect_files([library], recursive=True)}
    assert "deep.md" in deep and ".secret.md" not in deep


def test_ingest_then_search_with_citations(kb, library):
    outcomes = kb.ingest([library], collection="notes", recursive=True)
    assert {o.status for o in outcomes} == {"added"}
    assert all(o.chunks > 0 for o in outcomes)

    hits = kb.search("routing table prefixes", collection="notes", k=3)
    assert hits[0].source_name == "routers.md"
    assert hits[0].heading_path[0] == "Routers"
    assert "Routers" in hits[0].citation()

    pdf_hit = kb.search("firewall port filtering", k=1)[0]
    assert pdf_hit.source_name == "firewall.pdf" and pdf_hit.page_start == 1
    assert "p.1" in pdf_hit.citation()


def test_search_modes(kb, library):
    kb.ingest([library])
    keyword = kb.search("sourdough", mode="keyword")
    assert keyword and keyword[0].source_name == "baking.txt" and keyword[0].keyword_rank == 1
    vector = kb.search("sourdough starter flour", mode="vector")
    assert vector[0].source_name == "baking.txt" and vector[0].vector_score > 0
    with pytest.raises(ValueError):
        kb.search("x", mode="fuzzy")


def test_unchanged_files_skip_and_changed_files_replace(kb, library, embedder):
    kb.ingest([library])
    calls = len(embedder.document_calls)
    again = kb.ingest([library])
    assert {o.status for o in again} == {"unchanged"}
    assert len(embedder.document_calls) == calls  # nothing re-embedded

    (library / "baking.txt").write_text("Focaccia uses olive oil and rosemary.")
    outcomes = {o.path.rsplit("/", 1)[-1]: o.status for o in kb.ingest([library])}
    assert outcomes["baking.txt"] == "updated"
    assert not kb.search("sourdough", mode="keyword")  # old chunks and FTS rows gone
    assert kb.search("focaccia rosemary", mode="keyword")[0].source_name == "baking.txt"


def test_remove_cascades_to_chunks_and_keyword_index(kb, library, tmp_path):
    kb.ingest([library])
    assert kb.remove("baking.txt") == 1
    assert not kb.search("sourdough", mode="keyword")
    conn = sqlite3.connect(tmp_path / "kb.db")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('integrity-check')")
    orphans = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)"
    ).fetchone()[0]
    assert orphans == 0


def test_collections_are_pinned_to_their_embedder(tmp_path, library):
    db = tmp_path / "kb.db"
    with KnowledgeBase(db, embedder=HashingEmbedder(name="model-a")) as kb:
        kb.ingest([library / "baking.txt"], collection="a")
    with KnowledgeBase(db, embedder=HashingEmbedder(name="model-b")) as kb:
        with pytest.raises(EmbedderMismatchError):
            kb.ingest([library / "routers.md"], collection="a")
        with pytest.raises(EmbedderMismatchError):
            kb.search("bread", collection="a")
        # Unscoped search refuses rather than reporting "no results" for a library
        # that exists under another embedder
        with pytest.raises(EmbedderMismatchError, match="model-a"):
            kb.search("bread")
        kb.ingest([library / "routers.md"], collection="b")
        assert {c.name for c in kb.collections()} == {"a", "b"}
        assert kb.searchable_collections() == ["b"]  # skips "a" once something matches


def test_read_chunk_with_neighbors(kb, tmp_path):
    long_doc = tmp_path / "long.md"
    long_doc.write_text("# Long\n\n" + " ".join([LOREM] * 15))
    kb.config.chunking.max_tokens = 80
    kb.ingest([long_doc])
    assert kb.documents()[0].total_chunks >= 3
    first = kb.search("routing tables", k=1)[0].chunk_id
    key = first.rsplit("_chunk_", 1)[0]
    window = kb.read(f"{key}_chunk_002", neighbors=1)
    assert [h.chunk_id.rsplit("_", 1)[-1] for h in window] == ["001", "002", "003"]


def test_same_file_name_in_two_folders_gets_distinct_chunk_ids(kb, tmp_path):
    for folder, word in (("a", "alpha"), ("b", "bravo")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "README.md").write_text(f"# Readme\n\nThis readme is about {word}.")
    kb.ingest([tmp_path / "a" / "README.md", tmp_path / "b" / "README.md"])
    bravo = kb.search("bravo", mode="keyword", k=1)[0]
    alpha = kb.search("alpha", mode="keyword", k=1)[0]
    assert bravo.chunk_id != alpha.chunk_id
    assert "bravo" in kb.read(bravo.chunk_id, neighbors=0)[0].text


def test_mismatch_is_caught_before_any_processing(tmp_path, library):
    db = tmp_path / "kb.db"
    with KnowledgeBase(db, embedder=HashingEmbedder(name="model-a")) as kb:
        kb.ingest([library / "baking.txt"], collection="a")
    other = HashingEmbedder(name="model-b")
    with KnowledgeBase(db, embedder=other) as kb:
        with pytest.raises(EmbedderMismatchError):
            kb.ingest([library], collection="a")
    assert other.document_calls == []  # nothing extracted or embedded first


def test_retag_without_reembedding(kb, library, embedder):
    kb.ingest([library / "routers.md"], tags=["old"])
    calls = len(embedder.document_calls)
    [outcome] = kb.ingest([library / "routers.md"], tags=["networking", "lab"])
    assert outcome.status == "retagged" and len(embedder.document_calls) == calls
    assert sorted(kb.documents()[0].tags) == ["lab", "networking"]


def test_remove_accepts_relative_slash_paths(kb, library, monkeypatch):
    kb.ingest([library / "routers.md"])
    monkeypatch.chdir(library.parent)
    assert kb.remove("docs/routers.md") == 1


def test_failed_file_does_not_stop_the_batch(kb, library):
    (library / "broken.epub").write_bytes(b"not a zip")
    outcomes = {o.path.rsplit("/", 1)[-1]: o for o in kb.ingest([library])}
    assert outcomes["broken.epub"].status == "failed" and outcomes["broken.epub"].error
    assert outcomes["routers.md"].status == "added"


def test_tags_and_embedding_context(kb, library, embedder):
    kb.ingest([library / "routers.md"], tags=["networking"])
    assert kb.documents()[0].tags == ["networking"]
    embedded = embedder.document_calls[-1][0]
    assert embedded.startswith("routers.md › Routers")
    assert embedding_text("body", ["A", "B"], "f.md") == "f.md › A › B\n\nbody"


def test_fts_query_never_passes_syntax_through():
    assert fts_query('router" OR (NEAR x*') == '"router" OR "or" OR "near"'
    assert fts_query("?!") == ""
