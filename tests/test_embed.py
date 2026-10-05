"""Embedder specs and the Ollama/Aegis recipe, against a fake local Ollama."""

import builtins
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from autorag.embed import (
    EmbeddingUnavailableError,
    FastEmbedEmbedder,
    OllamaEmbedder,
    resolve_embedder,
)


@pytest.fixture
def fake_ollama():
    """Serves /api/embed, returning 768-dim vectors and recording requests."""
    received: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            vectors = [[float(i + 1)] * 768 for i, _ in enumerate(body["input"])]
            payload = json.dumps({"embeddings": vectors}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_aegis_preset_matches_aegis_recipe(fake_ollama, monkeypatch):
    url, received = fake_ollama
    monkeypatch.setenv("AEGIS_EMBED_URL", url)
    emb = resolve_embedder("aegis")
    docs = emb.embed_documents(["chunk one", "chunk two"])
    query = emb.embed_query("what is a router")

    assert received[0]["model"] == "nomic-embed-text"
    assert received[0]["input"] == ["search_document: chunk one", "search_document: chunk two"]
    assert received[1]["input"] == ["search_query: what is a router"]
    # Sliced to 512 and re-normalised, like Aegis's toUnit512
    assert docs.shape == (2, 512) and query.shape == (512,)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert emb.name == "aegis:nomic-embed-text@512"


def test_ollama_batches_and_spec_parsing(fake_ollama, monkeypatch):
    url, received = fake_ollama
    monkeypatch.setenv("AUTORAG_OLLAMA_URL", url)
    emb = resolve_embedder("ollama:mxbai-embed-large@256")
    assert isinstance(emb, OllamaEmbedder) and emb.dims == 256
    emb.batch_size = 2
    out = emb.embed_documents(["a", "b", "c"])
    assert out.shape == (3, 256) and len(received) == 2


def test_ollama_unreachable_raises_not_empty():
    emb = OllamaEmbedder("nomic-embed-text", url="http://127.0.0.1:9", timeout=2)
    with pytest.raises(EmbeddingUnavailableError):
        emb.embed_query("hello")


def test_resolve_defaults_and_errors(monkeypatch):
    monkeypatch.delenv("AUTORAG_EMBEDDER", raising=False)
    assert resolve_embedder().name == "fastembed:BAAI/bge-small-en-v1.5"
    monkeypatch.setenv("AUTORAG_EMBEDDER", "fastembed:BAAI/bge-base-en-v1.5")
    assert resolve_embedder().name == "fastembed:BAAI/bge-base-en-v1.5"
    with pytest.raises(ValueError):
        resolve_embedder("ollama")
    with pytest.raises(ValueError):
        resolve_embedder("openai:text-embedding-3")


def test_missing_fastembed_gives_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("no fastembed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EmbeddingUnavailableError, match=r"autorag\[embed\]"):
        FastEmbedEmbedder().embed_query("hi")
