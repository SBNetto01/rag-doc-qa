"""Embedders for the RAG pipeline.

Every embedder implements the same small interface used by the pipeline:

- ``embed_documents(texts) -> list[list[float]]``
- ``embed_query(text) -> list[float]``

The default **HashEmbedder** is dependency-free and deterministic: it maps
character n-grams to a fixed-size vector through a hash function, then L2
normalizes. It is intentionally simple — good enough to demo a full RAG loop
with **zero downloads and zero API keys**. For higher quality, install
``requirements-embeddings.txt`` and switch to ``SentenceTransformerEmbedder``
(a locally-run transformer model, still key-free).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

DEFAULT_DIM = 384


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9áéíóúâêôàãõç]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n_min: int, n_max: int):
    for n in range(n_min, n_max + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i : i + n])


class HashEmbedder:
    """Hashing n-gram embedder: no downloads, no keys, deterministic."""

    name = "offline-hash"

    def __init__(self, dim: int = DEFAULT_DIM, n_min: int = 1, n_max: int = 3) -> None:
        self.dim = dim
        self.n_min = n_min
        self.n_max = n_max

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for gram in _ngrams(_tokenize(text), self.n_min, self.n_max):
            digest = hashlib.md5(gram.encode("utf-8"), usedforsecurity=False).hexdigest()
            vector[int(digest[:8], 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers (local model, still key-free).

    Requires::

        pip install -r requirements-embeddings.txt
    """

    name = "local-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements-embeddings.txt"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(texts, normalize_embeddings=True)]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def build_embedder(kind: str) -> Embedder:
    """Instantiate an embedder by name (``offline`` | ``transformers``)."""
    if kind in ("offline", "offline-hash", "hash"):
        return HashEmbedder()
    if kind in ("transformers", "local-transformers", "sentence-transformers"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder '{kind}' (use 'offline' or 'transformers')")