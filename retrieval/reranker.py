"""Result re-ranking for improved retrieval quality."""

from __future__ import annotations

import logging

import numpy as np

from core.config import settings
from core.models import SearchResult

logger = logging.getLogger(__name__)


def rerank_cosine(
    query_embedding: list[float],
    results: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Re-rank results by cosine similarity between query and chunk embeddings.

    Args:
        query_embedding: Query embedding vector
        results: Initial search results to re-rank
        top_k: Number of top results to return

    Returns:
        Re-ranked search results with updated scores and ranks
    """
    if not results:
        return []

    # Filter out results without embeddings
    valid_results = [r for r in results if r.chunk.embedding]
    if not valid_results:
        logger.warning("No results with embeddings to rerank")
        return results[:top_k]

    # Convert to numpy arrays for efficient computation
    query_vec = np.array(query_embedding)
    query_norm = np.linalg.norm(query_vec)

    if query_norm == 0:
        logger.warning("Query embedding has zero norm, returning original results")
        return results[:top_k]

    # Compute cosine similarity for each result
    scored_results = []
    for result in valid_results:
        chunk_vec = np.array(result.chunk.embedding)
        chunk_norm = np.linalg.norm(chunk_vec)

        if chunk_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_vec, chunk_vec) / (query_norm * chunk_norm))

        # Create new SearchResult with updated score
        scored_results.append(
            SearchResult(chunk=result.chunk, score=similarity, rank=result.rank)
        )

    # Sort by similarity descending
    scored_results.sort(key=lambda r: r.score, reverse=True)

    # Take top_k and update ranks
    reranked = []
    for i, result in enumerate(scored_results[:top_k]):
        reranked.append(
            SearchResult(chunk=result.chunk, score=result.score, rank=i + 1)
        )

    logger.debug("Reranked %d results to top %d", len(scored_results), len(reranked))
    return reranked


def rerank(
    query: str,
    query_embedding: list[float],
    results: list[SearchResult],
    top_k: int | None = None,
    method: str | None = None,
) -> list[SearchResult]:
    """Re-rank search results using specified method.

    Args:
        query: Original query text (for cross-encoder methods)
        query_embedding: Query embedding vector
        results: Search results to re-rank
        top_k: Number of top results to return (default: settings.top_k_rerank)
        method: Re-ranking method to use (default: settings.rerank_method)

    Returns:
        Re-ranked search results with updated scores and ranks
    """
    if top_k is None:
        top_k = settings.top_k_rerank

    if method is None:
        method = settings.rerank_method

    if method == "cosine":
        return rerank_cosine(query_embedding, results, top_k)
    else:
        logger.warning("Unknown rerank method '%s', using cosine", method)
        return rerank_cosine(query_embedding, results, top_k)
