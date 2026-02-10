"""Data models for RAG 2.0 pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk with optional contextual enrichment and embedding."""

    id: str = ""
    content: str
    context: str = ""
    embedding: list[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @property
    def enriched_content(self) -> str:
        if self.context:
            return f"{self.context}\n\n{self.content}"
        return self.content


class SearchResult(BaseModel):
    """A single search result from vector store."""

    chunk: Chunk
    score: float = 0.0
    rank: int = 0


class QAResult(BaseModel):
    """Final Q&A result with answer, sources, and confidence."""

    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    confidence: float = 0.0
    query: str = ""
    expanded_query: str = ""
    retries: int = 0
