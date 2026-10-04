"""API tests using FastAPI's TestClient (lifespan + offline embedder)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.embedders import HashEmbedder
from app.pipeline import RagPipeline
from app import main


def _set_index(client: TestClient, index_dir: Path) -> None:
    client.app.state.pipeline = RagPipeline(HashEmbedder(), index_dir=index_dir)


def test_health_before_ingest(tmp_path: Path) -> None:
    with TestClient(main.app) as client:
        _set_index(client, tmp_path / "empty")
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["index_present"] is False
        assert response.json()["embedder"] == "offline-hash"


def test_ingest_and_ask_flow(tmp_path: Path) -> None:
    with TestClient(main.app) as client:
        _set_index(client, tmp_path / "index")

        ingest = client.post(
            "/ingest",
            json={
                "documents": [
                    {"filename": "faq.txt", "text": "The capital of RAG land is Retrieval City.\n"
                                                   "Discounts apply after ten thousand purchases."}
                ]
            },
        )
        assert ingest.status_code == 200
        assert ingest.json()["chunks"] >= 1

        health = client.get("/health")
        assert health.json()["index_present"] is True

        ask = client.post("/ask", json={"question": "what is the capital?", "k": 2})
        assert ask.status_code == 200
        body = ask.json()
        assert body["sources"], "expected retrieved sources"
        assert body["sources"][0]["source"] == "faq.txt"
        assert "retrieval city" in body["answer"].lower()
        assert body["generated"] is False


def test_ask_without_index_returns_409(tmp_path: Path) -> None:
    with TestClient(main.app) as client:
        _set_index(client, tmp_path / "missing")
        response = client.post("/ask", json={"question": "hi there"})
        assert response.status_code == 409
        assert "ingest" in response.json()["detail"].lower()


def test_cannot_ingest_empty_documents(tmp_path: Path) -> None:
    with TestClient(main.app) as client:
        _set_index(client, tmp_path / "index")
        response = client.post("/ingest", json={"documents": []})
        assert response.status_code == 422