"""Local embedding backends.

An embedder is named by a spec string, which is also what a collection records so
it is never searched with vectors from a different model:

- ``fastembed`` / ``fastembed:<model>`` — ONNX on CPU via the optional ``[embed]``
  extra. Default model ``BAAI/bge-small-en-v1.5`` (384 dims), the same default
  MonkChat ships. Downloads the model once, offline afterwards.
- ``ollama:<model>`` — a running Ollama server's ``/api/embed``; no Python deps.
- ``aegis`` — Ollama ``nomic-embed-text`` with Aegis's exact recipe
  (``search_document: `` / ``search_query: `` prefixes, sliced to 512 dims and
  re-normalised), so vectors are interchangeable with Aegis's pgvector columns.

Vectors are always returned L2-normalised as float32, so cosine similarity is a
dot product.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol, Sequence

import numpy as np

DEFAULT_SPEC = "fastembed"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class EmbeddingUnavailableError(RuntimeError):
    """The embedding backend can't be reached or isn't installed.

    Raised rather than returning nothing, so "no embedder" never reads as "no
    results" (the same lesson Aegis learned).
    """


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FastEmbedEmbedder:
    def __init__(self, model: str = DEFAULT_FASTEMBED_MODEL) -> None:
        self.model = model
        self.name = f"fastembed:{model}"
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "fastembed is not installed. Install it with: pip install 'autorag[embed]' "
                    "— or use an Ollama embedder (--embedder ollama:nomic-embed-text)."
                ) from exc
            try:
                self._engine = TextEmbedding(self.model)
            except Exception as exc:
                raise EmbeddingUnavailableError(f"Could not load embedding model {self.model}: {exc}") from exc
        return self._engine

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return normalize(np.array(list(self._load().passage_embed(list(texts)))))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize(np.array(list(self._load().query_embed(text))))[0]


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        url: str | None = None,
        document_prefix: str = "",
        query_prefix: str = "",
        dims: int | None = None,
        name: str | None = None,
        batch_size: int = 32,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.url = (url or os.environ.get("AUTORAG_OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/")
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.dims = dims
        self.batch_size = batch_size
        self.timeout = timeout
        self.name = name or f"ollama:{model}" + (f"@{dims}" if dims else "")

    def _post(self, inputs: list[str]) -> np.ndarray:
        body = json.dumps({"model": self.model, "input": inputs}).encode()
        request = urllib.request.Request(
            f"{self.url}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise EmbeddingUnavailableError(f"Ollama rejected the embed request ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EmbeddingUnavailableError(f"Ollama is not reachable at {self.url}: {exc}") from exc
        embeddings = payload.get("embeddings")
        if not embeddings or len(embeddings) != len(inputs):
            raise EmbeddingUnavailableError(f"Ollama returned no embeddings for model {self.model}")
        vectors = np.asarray(embeddings, dtype=np.float32)
        if self.dims:
            if vectors.shape[1] < self.dims:
                raise EmbeddingUnavailableError(
                    f"{self.model} returns {vectors.shape[1]} dims, fewer than the {self.dims} requested"
                )
            # Matryoshka truncation, then re-normalise (what Aegis's toUnit512 does)
            vectors = vectors[:, : self.dims]
        return normalize(vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        batches = [
            self._post([self.document_prefix + t for t in texts[i : i + self.batch_size]])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        return self._post([self.query_prefix + text])[0]


def resolve_embedder(spec: str | None = None) -> Embedder:
    """Build an embedder from a spec string (see module docstring).

    Falls back to ``$AUTORAG_EMBEDDER``, then ``fastembed``.
    """
    spec = (spec or os.environ.get("AUTORAG_EMBEDDER") or DEFAULT_SPEC).strip()
    kind, _, arg = spec.partition(":")
    kind = kind.lower()
    if kind == "fastembed":
        return FastEmbedEmbedder(arg or DEFAULT_FASTEMBED_MODEL)
    if kind == "ollama":
        if not arg:
            raise ValueError("An Ollama embedder needs a model, e.g. ollama:nomic-embed-text")
        model, _, dims = arg.partition("@")
        return OllamaEmbedder(model, dims=int(dims) if dims else None)
    if kind == "aegis":
        return OllamaEmbedder(
            "nomic-embed-text",
            url=os.environ.get("AEGIS_EMBED_URL") or None,
            document_prefix="search_document: ",
            query_prefix="search_query: ",
            dims=512,
            name="aegis:nomic-embed-text@512",
        )
    raise ValueError(f"Unknown embedder {spec!r}. Use fastembed[:model], ollama:<model>[@dims], or aegis.")
