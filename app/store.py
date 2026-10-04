"""Local vector store built on numpy.

A small cosine-similarity store with on-disk persistence, kept dependency-free
(on top of numpy) so the full RAG loop runs anywhere — including this repo's
tests — without native binaries. For very large corpora, swap it for FAISS or
a real vector database (see README -> Production notes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    """In-memory vectors with numpy cosine search and npz/json persistence."""

    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._texts: list[str] = []
        self._metadatas: list[dict] = []

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, vectors: list[list[float]], texts: list[str], metadatas: list[dict]) -> None:
        self._vectors.extend(np.asarray(vector, dtype=np.float32) for vector in vectors)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, dict, float]]:
        """Return ``(text, metadata, score)`` best first; score = cosine similarity (0..1)."""
        if not self._vectors:
            return []
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm
        scores = [
            float(np.dot(query, vector.ravel() / (float(np.linalg.norm(vector)) or 1.0)))
            for vector in self._vectors
        ]
        order = np.argsort(scores)[::-1][:k]
        return [(self._texts[i], self._metadatas[i], round(scores[i], 4)) for i in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._vectors:
            matrix = np.vstack(self._vectors)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)
        np.savez_compressed(directory / "vectors.npz", matrix=matrix)
        (directory / "meta.json").write_text(
            json.dumps(
                [{"text": text, "metadata": meta} for text, meta in zip(self._texts, self._metadatas)],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        store = cls()
        vectors_file = directory / "vectors.npz"
        meta_file = directory / "meta.json"
        if not (vectors_file.exists() and meta_file.exists()):
            return store
        matrix = np.load(str(vectors_file))["matrix"]
        entries = json.loads(meta_file.read_text(encoding="utf-8"))
        store._vectors = [np.asarray(row, dtype=np.float32) for row in matrix]
        store._texts = [entry["text"] for entry in entries]
        store._metadatas = [entry["metadata"] for entry in entries]
        return store

    def empty(self) -> bool:
        return not self._texts