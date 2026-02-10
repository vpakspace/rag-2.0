"""Main retrieval orchestrator with query expansion and re-ranking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.config import settings
from core.models import SearchResult
from retrieval.query_expander import expand_query, generate_multi_queries
from retrieval.reranker import rerank

if TYPE_CHECKING:
    from openai import OpenAI

    from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates retrieval pipeline: expansion -> multi-query -> search -> rerank."""

    def __init__(self, vector_store: VectorStore, openai_client: OpenAI | None = None):
        """Initialize retriever with vector store and optional OpenAI client.

        Args:
            vector_store: Vector store for similarity search
            openai_client: Optional OpenAI client (will create if None)
        """
        self.vector_store = vector_store

        if openai_client is None:
            from openai import OpenAI

            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = openai_client

    def get_embedding(self, text: str) -> list[float]:
        """Embed text using OpenAI embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        response = self.openai_client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Execute full retrieval pipeline with expansion and re-ranking.

        Pipeline steps:
        1. Expand query using LLM
        2. Generate multi-query variations
        3. Embed all queries
        4. Search vector store for each query
        5. Merge and deduplicate results
        6. Re-rank merged results
        7. Return top_k results

        Args:
            query: User query
            top_k: Number of final results to return (default: settings.top_k_rerank)

        Returns:
            Re-ranked search results
        """
        if top_k is None:
            top_k = settings.top_k_rerank

        # Step 1: Expand query
        expanded = expand_query(query, self.openai_client)
        logger.info("Expanded query: %s", expanded)

        # Step 2: Generate multi-queries
        multi_queries = generate_multi_queries(expanded, n=3, openai_client=self.openai_client)
        all_queries = [expanded] + multi_queries
        logger.info("Generated %d total queries", len(all_queries))

        # Step 3: Embed all queries
        query_embeddings = [self.get_embedding(q) for q in all_queries]
        primary_query_embedding = query_embeddings[0]  # Expanded query embedding

        # Step 4: Search for each query
        all_results: list[SearchResult] = []
        for i, query_emb in enumerate(query_embeddings):
            results = self.vector_store.search(
                query_emb, top_k=settings.top_k_retrieval
            )
            logger.debug("Query %d returned %d results", i, len(results))
            all_results.extend(results)

        # Step 5: Deduplicate by chunk.id (keep highest score)
        if not all_results:
            logger.warning("No results found for query: %s", query)
            return []

        dedup_map: dict[str, SearchResult] = {}
        for result in all_results:
            chunk_id = result.chunk.id
            if chunk_id not in dedup_map or result.score > dedup_map[chunk_id].score:
                dedup_map[chunk_id] = result

        merged_results = list(dedup_map.values())
        logger.info(
            "Merged %d results into %d unique chunks",
            len(all_results),
            len(merged_results),
        )

        # Step 6: Re-rank merged results
        reranked = rerank(
            query=query,
            query_embedding=primary_query_embedding,
            results=merged_results,
            top_k=top_k,
        )

        logger.info("Retrieved and reranked to %d final results", len(reranked))
        return reranked
