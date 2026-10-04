"""Pipeline tests: ingestion and retrieval with the offline hash embedder.

No API keys and no model downloads — the hash embedder is deterministic,
so the assertions below are stable across runs.
"""

from __future__ import annotations

from app.embedders import HashEmbedder
from app.pipeline import RagPipeline


def _pipeline(index_dir):
    return RagPipeline(HashEmbedder(), index_dir=index_dir)


def test_ingest_creates_index_and_chunks(index_dir, corpus) -> None:
    pipeline = _pipeline(index_dir)
    summary = pipeline.ingest_texts(corpus)

    assert summary["documents"] == 2
    assert summary["chunks"] >= 2
    assert pipeline.index_exists()
    assert (index_dir / "vectors.npz").exists()
    assert (index_dir / "meta.json").exists()


def test_ask_returns_top_matching_source(index_dir, corpus) -> None:
    pipeline = _pipeline(index_dir)
    pipeline.ingest_texts(corpus)

    result = pipeline.ask("when are salaries paid?", k=3)

    assert result.sources, "expected at least one source"
    assert result.sources[0].source == "handbook.md"
    assert "salaries are paid" in result.answer.lower()
    assert all(0 <= s.score for s in result.sources)


def test_ask_without_index_returns_friendly_message(index_dir) -> None:
    pipeline = _pipeline(index_dir)

    result = pipeline.ask("anything")

    assert result.sources == []
    assert "ingest" in result.answer.lower()


def test_ingest_second_time_rebuilds_index(index_dir, corpus) -> None:
    pipeline = _pipeline(index_dir)
    pipeline.ingest_texts(corpus)
    first = pipeline.index_exists()

    summary = pipeline.ingest_texts([("extra.txt", "A brand new paragraph to embed.")])

    assert first is True
    assert summary["documents"] == 1
    assert pipeline.index_exists()