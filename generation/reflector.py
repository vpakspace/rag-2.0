"""Self-reflective evaluation and retry query generation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.config import settings
from core.models import QAResult, SearchResult
from generation.generator import generate_answer

if TYPE_CHECKING:
    from openai import OpenAI

    from retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


def evaluate_relevance(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None
) -> float:
    """Evaluate how relevant retrieved chunks are to the query.

    Args:
        query: User query
        results: Retrieved search results
        openai_client: Optional OpenAI client (will create if None)

    Returns:
        Average relevance score (1-5 scale)
    """
    # Initialize OpenAI client if not provided
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=settings.openai_api_key)

    # Handle empty results
    if not results:
        logger.warning("No results to evaluate")
        return 0.0

    # Build context from chunks
    context_chunks = []
    for i, result in enumerate(results, start=1):
        chunk_text = result.chunk.enriched_content
        context_chunks.append(f"[Chunk {i}]\n{chunk_text[:300]}...")  # Truncate for evaluation

    context = "\n\n".join(context_chunks)

    # Build prompt
    prompt = f"""Rate the relevance of each retrieved chunk to the query on a scale of 1-5.
Return ONLY a comma-separated list of scores.

Query: {query}

Chunks:
{context}

Scores (comma-separated):"""

    try:
        response = openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        scores_text = response.choices[0].message.content or ""
        logger.debug("Relevance scores response: %s", scores_text)

        # Parse scores
        scores = []
        for s in scores_text.split(","):
            try:
                score = float(s.strip())
                scores.append(score)
            except ValueError:
                logger.warning("Invalid score in response: %s", s)
                # Use default score 2.5 for unparseable scores
                scores.append(2.5)

        # If we got fewer scores than chunks, pad with 2.5
        while len(scores) < len(results):
            scores.append(2.5)

        # Calculate average
        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info("Average relevance score: %.2f", avg_score)
        return avg_score

    except Exception as e:
        logger.error("Error evaluating relevance: %s", e)
        return 2.5  # Default middle score on error


def generate_retry_query(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None
) -> str:
    """Generate an improved query based on what was found (and what was missing).

    Args:
        query: Original query
        results: Retrieved search results
        openai_client: Optional OpenAI client (will create if None)

    Returns:
        Improved query string
    """
    # Initialize OpenAI client if not provided
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=settings.openai_api_key)

    # Build summary of results
    if results:
        summary_parts = []
        for i, result in enumerate(results[:3], start=1):  # Only top 3 for summary
            chunk_text = result.chunk.enriched_content[:200]
            summary_parts.append(f"{i}. {chunk_text}...")
        summary = "\n".join(summary_parts)
    else:
        summary = "No relevant content found."

    # Build prompt
    prompt = f"""The search didn't find good results. Original query: {query}

Found content:
{summary}

Generate a better search query that might find more relevant information."""

    try:
        response = openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Higher temperature for more creative reformulations
        )

        retry_query = response.choices[0].message.content or query
        logger.info("Generated retry query: %s", retry_query)
        return retry_query

    except Exception as e:
        logger.error("Error generating retry query: %s", e)
        return query  # Fall back to original query


def reflect_and_answer(
    query: str, retriever: Retriever, openai_client: OpenAI | None = None
) -> QAResult:
    """Main orchestrator: retrieve -> evaluate -> retry if needed -> answer.

    Pipeline:
    1. retriever.retrieve(query) -> results
    2. evaluate_relevance(query, results) -> avg_score
    3. If avg_score >= settings.relevance_threshold (3.0): generate_answer(query, results)
    4. Else: generate_retry_query -> retriever.retrieve(retry_query) -> evaluate again
    5. Max settings.max_retries (2) retry cycles
    6. After retries exhausted, generate_answer with best results found
    7. Set QAResult.retries to number of retries used

    Args:
        query: User query
        retriever: Retriever instance for search
        openai_client: Optional OpenAI client (will create if None)

    Returns:
        QAResult with answer, sources, confidence, and retries count
    """
    # Initialize OpenAI client if not provided
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=settings.openai_api_key)

    # Track best results and scores
    best_results = []
    best_score = 0.0
    current_query = query
    retries_used = 0

    for retry_count in range(settings.max_retries + 1):  # +1 for initial attempt
        logger.info("Attempt %d: Retrieving for query: %s", retry_count + 1, current_query)

        # Step 1: Retrieve
        results = retriever.retrieve(current_query)

        # Step 2: Evaluate relevance
        avg_score = evaluate_relevance(current_query, results, openai_client)

        # Track best results
        if avg_score > best_score or not best_results:
            best_results = results
            best_score = avg_score

        # Step 3: Check if good enough
        if avg_score >= settings.relevance_threshold:
            logger.info(
                "Relevance threshold met (%.2f >= %.2f)",
                avg_score,
                settings.relevance_threshold,
            )
            break

        # Step 4: Retry if attempts remaining
        if retry_count < settings.max_retries:
            logger.info(
                "Relevance too low (%.2f < %.2f), generating retry query",
                avg_score,
                settings.relevance_threshold,
            )
            current_query = generate_retry_query(query, results, openai_client)
            retries_used += 1
        else:
            logger.warning(
                "Max retries reached (%d), using best results found", settings.max_retries
            )

    # Step 5: Generate answer with best results
    qa_result = generate_answer(query, best_results, openai_client)
    qa_result.retries = retries_used
    qa_result.expanded_query = current_query if retries_used > 0 else ""
    qa_result.confidence = best_score / 5.0  # Normalize to 0-1

    return qa_result
