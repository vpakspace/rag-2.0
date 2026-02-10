"""Agent tools for RAG 2.0 pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.config import settings

if TYPE_CHECKING:
    from openai import OpenAI

    from retrieval.retriever import Retriever
    from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def vector_search(query: str, retriever: Retriever, top_k: int = 5) -> list[dict]:
    """Quick vector search — embed query and search store.

    Returns list of dicts with content, score, id.
    """
    embedding = retriever.get_embedding(query)
    results = retriever.vector_store.search(embedding, top_k=top_k)
    return [
        {"id": r.chunk.id, "content": r.chunk.enriched_content, "score": r.score}
        for r in results
    ]


def full_document_read(vector_store: VectorStore) -> str:
    """Read all chunks from vector store concatenated as full document."""
    from storage.vector_store import NODE_LABEL

    with vector_store._driver.session() as session:
        result = session.run(
            f"""
            MATCH (c:{NODE_LABEL})
            RETURN c.content AS content, c.metadata AS metadata
            ORDER BY c.id
            """
        )
        parts = [record["content"] for record in result if record["content"]]

    return "\n\n".join(parts)


def focused_search(
    query: str, retriever: Retriever, top_k: int = 5
) -> list[dict]:
    """Deep search with query expansion + re-ranking.

    Uses full Retriever pipeline: expand → multi-query → search → rerank.
    Returns list of dicts with content, score, rank.
    """
    results = retriever.retrieve(query, top_k=top_k)
    return [
        {
            "id": r.chunk.id,
            "content": r.chunk.enriched_content,
            "score": r.score,
            "rank": r.rank,
        }
        for r in results
    ]
