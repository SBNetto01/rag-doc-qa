"""Request/response models for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    filename: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=50)


class IngestResponse(BaseModel):
    chunks: int
    documents: int


class SourceOut(BaseModel):
    source: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)
    use_llm: bool = False


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    generated: bool
    generation: str | None = None


class HealthResponse(BaseModel):
    status: str
    index_present: bool
    embedder: str