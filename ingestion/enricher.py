"""Contextual enrichment and embedding for chunks via LLM."""

from __future__ import annotations

import logging
import time

import openai

from core.config import settings
from core.models import Chunk

logger = logging.getLogger(__name__)


def enrich_chunks(
    chunks: list[Chunk], document_summary: str = ""
) -> list[Chunk]:
    """Enrich chunks with contextual information via LLM.

    If no document_summary provided, generates one from first few chunks.
    For each chunk, calls OpenAI to generate 1-2 sentence context explaining
    the chunk's role within the document.

    Sets chunk.context = LLM response.
    """
    if not chunks:
        return chunks

    client = openai.OpenAI(api_key=settings.openai_api_key)

    # Generate document summary if not provided
    if not document_summary:
        document_summary = _generate_summary(chunks[:3], client)
        logger.info("Generated document summary: %s", document_summary[:100])

    # Enrich each chunk
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        try:
            context = _generate_context(chunk.content, document_summary, client)
            chunk.context = context
            logger.debug(
                "Enriched chunk %d/%d: %s",
                i + 1,
                len(chunks),
                context[:50],
            )

            # Rate limiting - small sleep between calls
            if i < len(chunks) - 1:
                time.sleep(0.1)

            enriched_chunks.append(chunk)

        except Exception as e:
            logger.warning("Failed to enrich chunk %d: %s", i, e)
            enriched_chunks.append(chunk)

    logger.info("Enriched %d chunks", len(enriched_chunks))
    return enriched_chunks


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Batch embed chunks using OpenAI embeddings API.

    Sets chunk.embedding for each chunk.
    """
    if not chunks:
        return chunks

    client = openai.OpenAI(api_key=settings.openai_api_key)

    # Prepare texts to embed (use enriched_content if available)
    texts = [chunk.enriched_content for chunk in chunks]

    try:
        response = client.embeddings.create(
            model=settings.embedding_model, input=texts
        )

        # Set embeddings
        for i, chunk in enumerate(chunks):
            chunk.embedding = response.data[i].embedding

        logger.info("Embedded %d chunks", len(chunks))

    except Exception as e:
        logger.error("Failed to embed chunks: %s", e)
        raise

    return chunks


def _generate_summary(chunks: list[Chunk], client: openai.OpenAI) -> str:
    """Generate document summary from first few chunks."""
    combined_text = "\n\n".join(chunk.content for chunk in chunks)

    prompt = f"""Here are the first few sections of a document:

{combined_text[:2000]}

Write 2-3 sentences summarizing what this document is about."""

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content or "Unknown document"
    except Exception as e:
        logger.warning("Failed to generate summary: %s", e)
        return "Document"


def _generate_context(
    chunk_content: str, document_summary: str, client: openai.OpenAI
) -> str:
    """Generate 1-2 sentence context for a chunk."""
    prompt = f"""Here's the document: {document_summary}

Here's a chunk from the document:

{chunk_content[:500]}

Write 1-2 sentences explaining the context of this chunk within the document. Be specific and concise."""

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Failed to generate context: %s", e)
        return ""
