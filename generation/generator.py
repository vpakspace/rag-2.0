"""LLM answer generation from retrieved chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.config import settings
from core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_answer(
    query: str, results: list[SearchResult], openai_client: OpenAI | None = None
) -> QAResult:
    """Generate answer from query and retrieved chunks using LLM.

    Args:
        query: User query
        results: Retrieved search results (chunks with scores)
        openai_client: Optional OpenAI client (will create if None)

    Returns:
        QAResult with answer, sources, and confidence
    """
    # Initialize OpenAI client if not provided
    if openai_client is None:
        from openai import OpenAI

        openai_client = OpenAI(api_key=settings.openai_api_key)

    # Handle empty results
    if not results:
        logger.warning("No results provided for answer generation")
        return QAResult(
            answer="I don't have enough context to answer this question.",
            sources=[],
            confidence=0.0,
            query=query,
            expanded_query="",
            retries=0,
        )

    # Build context from chunks
    context_chunks = []
    for i, result in enumerate(results, start=1):
        # Use enriched_content if available, otherwise content
        chunk_text = result.chunk.enriched_content
        context_chunks.append(f"[Chunk {i}]\n{chunk_text}")

    context = "\n\n".join(context_chunks)

    # Build prompt
    system_prompt = (
        "You are a precise Q&A assistant. Answer ONLY based on the provided context. "
        "If the context doesn't contain enough information, say so. "
        "Always cite which chunk(s) your answer is based on."
    )

    user_prompt = f"""Query: {query}

Context:
{context}

Please provide an answer based on the above context. Cite which chunk(s) you used."""

    # Call LLM
    logger.info("Generating answer for query: %s", query)
    try:
        response = openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Lower temperature for more precise answers
        )

        answer_text = response.choices[0].message.content or ""

        # Extract confidence if present (default 0.5)
        # LLM might include confidence in response, but for now use default
        confidence = 0.5

        logger.info("Generated answer: %s", answer_text[:100])

        return QAResult(
            answer=answer_text,
            sources=results,
            confidence=confidence,
            query=query,
            expanded_query="",
            retries=0,
        )

    except Exception as e:
        logger.error("Error generating answer: %s", e)
        return QAResult(
            answer=f"Error generating answer: {str(e)}",
            sources=results,
            confidence=0.0,
            query=query,
            expanded_query="",
            retries=0,
        )
