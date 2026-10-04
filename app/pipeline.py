"""In-memory RAG pipeline: chunk → embed → index → retrieve → answer.

The pipeline runs fully offline (local embeddings, local vector store on disk)
so the project works without any API key. Generation over the retrieved context
is optional and delegated to a local LLM server such as Ollama.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embedders import Embedder
from .loaders import load_document
from .store import VectorStore


@dataclass
class SourceHit:
    source: str
    text: str
    score: float


@dataclass
class Answer:
    answer: str
    sources: list[SourceHit] = field(default_factory=list)
    generation: str | None = None
    generated: bool = False


class RagPipeline:
    """Ingest documents and answer questions against a local vector index."""

    def __init__(
        self,
        embedder: Embedder,
        index_dir: str | Path = "data/index",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ---- indexing ------------------------------------------------------

    def ingest_texts(self, documents: list[tuple[str, str]]) -> dict:
        """Index ``(filename, text)`` pairs. Returns a summary dict."""
        chunks_text: list[str] = []
        chunks_meta: list[dict] = []
        for filename, text in documents:
            for chunk in self._splitter.split_text(text):
                chunks_text.append(chunk)
                chunks_meta.append({"source": filename})

        if not chunks_text:
            return {"chunks": 0, "documents": len(documents)}

        vectors = self.embedder.embed_documents(chunks_text)
        store = VectorStore()
        store.add(vectors, chunks_text, chunks_meta)
        store.save(self.index_dir)
        return {"chunks": len(chunks_text), "documents": len(documents)}

    def ingest_files(self, paths: list[str | Path]) -> dict:
        documents: list[tuple[str, str]] = []
        for raw in paths:
            path = Path(raw)
            documents.append((path.name, load_document(path)))
        return self.ingest_texts(documents)

    def index_exists(self) -> bool:
        return (self.index_dir / "vectors.npz").exists() and (self.index_dir / "meta.json").exists()

    # ---- retrieval -----------------------------------------------------

    def ask(self, question: str, k: int = 4, use_llm: bool = False) -> Answer:
        if not self.index_exists():
            return Answer(
                answer="No documents have been ingested yet - run `ragdoc ingest ./docs` first.",
            )

        store = VectorStore.load(self.index_dir)
        query_vector = self.embedder.embed_query(question)
        hits = [
            SourceHit(
                source=metadata.get("source", "unknown"),
                text=text.strip(),
                score=score,
            )
            for text, metadata, score in store.search(query_vector, k=k)
        ]
        if not hits:
            return Answer(answer="No relevant content found.", sources=[])

        answer = Answer(answer=self._compose_answer(question, hits), sources=hits)
        if use_llm:
            answer.generation = self._generate_with_ollama(question, hits)
            answer.generated = answer.generation is not None
        return answer

    @staticmethod
    def _compose_answer(question: str, hits: list[SourceHit]) -> str:
        best = hits[0]
        return f"Best match from '{best.source}':\n{_matching_sentences(best.text, question)}"

    # ---- optional local LLM generation ----------------------------------

    def _generate_with_ollama(self, question: str, hits: list[SourceHit], top_n: int = 3) -> str | None:
        """Ask a local Ollama server to answer from the retrieved context.

        Never requires an API key — only a running Ollama instance
        (default RAG_OLLAMA_BASE_URL=http://localhost:11434).
        """
        base_url = os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("RAG_OLLAMA_MODEL", "llama3.2")
        context = "\n\n".join(f"[{h.source}] {h.text[:1200]}" for h in hits[:top_n])
        prompt = (
            f"Answer the question using ONLY the context below. "
            f"If the context does not contain the answer, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        try:
            import httpx

            response = httpx.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=httpx.Timeout(120.0),
            )
            response.raise_for_status()
            payload = response.json()
            return str(payload.get("response", "")).strip() or None
        except Exception as exc:  # noqa: BLE001 - fall back to extractive answer
            return f"(could not reach Ollama at {base_url}: {exc})"


def _matching_sentences(text: str, question: str, max_sentences: int = 2) -> str:
    """Return the chunk sentences most related to the question (keyword overlap)."""
    lines = [line.strip() for line in text.splitlines()]
    kept = [
        line
        for line in lines
        if line
        and not line.lstrip().startswith(("#", ">", "`"))
        and "ragdoc ingest" not in line.lower()
    ]
    body = " ".join(kept) if kept else text
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", body) if len(s.strip()) > 20]

    question_words = {
        w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2
    }
    if not question_words:
        return body[:280]

    def overlap(sentence: str) -> int:
        words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        return len(words & question_words)

    ranked = sorted(sentences, key=overlap, reverse=True)
    chosen = [s for s in ranked if overlap(s) > 0][:max_sentences] or ranked[:1]
    return " ".join(chosen)[:340]


def dump_json(payload: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")