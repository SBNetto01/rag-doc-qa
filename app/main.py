"""FastAPI service for key-free RAG question answering over your documents.

Run with::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .embedders import Embedder, build_embedder
from .pipeline import Answer, RagPipeline
from .schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceOut,
)

INDEX_DIR = os.getenv("RAG_INDEX_DIR", "data/index")
EMBEDDER_KIND = os.getenv("RAG_EMBEDDER", "offline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = build_embedder(EMBEDDER_KIND)
    app.state.pipeline = RagPipeline(app.state.embedder, index_dir=INDEX_DIR)
    yield


app = FastAPI(
    title="rag-doc-qa",
    version="0.1.0",
    description=(
        "Key-free RAG question answering over your own documents. "
        "Local embeddings (offline by default), optional local LLM generation via Ollama."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> dict:
    return {
        "name": "rag-doc-qa",
        "docs": "/docs",
        "health": "/health",
        "ingest": "POST /ingest",
        "ask": "POST /ask",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pipeline: RagPipeline = app.state.pipeline
    return HealthResponse(
        status="ok",
        index_present=pipeline.index_exists(),
        embedder=pipeline.embedder.name,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    pipeline: RagPipeline = app.state.pipeline
    summary = pipeline.ingest_texts(
        [(doc.filename, doc.text) for doc in payload.documents]
    )
    return IngestResponse(chunks=summary["chunks"], documents=summary["documents"])


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    pipeline: RagPipeline = app.state.pipeline
    if not pipeline.index_exists():
        raise HTTPException(
            status_code=409,
            detail="No index found — ingest documents via POST /ingest first.",
        )
    result: Answer = pipeline.ask(payload.question, k=payload.k, use_llm=payload.use_llm)
    return AskResponse(
        question=payload.question,
        answer=result.answer,
        sources=[SourceOut(source=h.source, score=h.score) for h in result.sources],
        generated=result.generated,
        generation=result.generation,
    )