"""Semantic text chunker with markdown-aware splitting."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from core.config import settings
from core.models import Chunk

if TYPE_CHECKING:
    pass


def chunk_text(
    text: str, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> list[Chunk]:
    """Chunk text semantically using markdown structure.

    Strategy:
    1. Split by markdown headers (##, ###) first
    2. Then by paragraphs (\\n\\n)
    3. If still too large, split by sentences
    4. Tables (lines starting with |) kept as atomic units

    Each chunk gets:
    - Auto-generated id (md5 of content)
    - metadata: section_title (from nearest header), chunk_index
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if chunk_overlap is None:
        chunk_overlap = settings.chunk_overlap

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    current_section = ""

    # Split by markdown headers first
    sections = _split_by_headers(text)

    for section_title, section_content in sections:
        # Process this section
        section_chunks = _chunk_section(
            section_content, chunk_size, chunk_overlap, section_title
        )
        chunks.extend(section_chunks)
        current_section = section_title

    # Add chunk_index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split text by markdown headers (## , ### ).

    Returns list of (section_title, section_content) tuples.
    """
    # Pattern for markdown headers (## or ###)
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str, str]] = []
    current_title = ""
    current_content: list[str] = []

    for line in text.split("\n"):
        match = header_pattern.match(line)
        if match:
            # Save previous section
            if current_content:
                sections.append((current_title, "\n".join(current_content)))

            # Start new section
            current_title = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_content:
        sections.append((current_title, "\n".join(current_content)))

    # If no sections found, return entire text as one section
    if not sections:
        sections.append(("", text))

    return sections


def _chunk_section(
    text: str, chunk_size: int, chunk_overlap: int, section_title: str
) -> list[Chunk]:
    """Chunk a single section into chunks."""
    if not text.strip():
        return []

    chunks: list[Chunk] = []

    # Check if section is a table (lines starting with |)
    lines = text.split("\n")
    is_table = all(
        line.strip().startswith("|") or not line.strip() for line in lines
    )

    if is_table and len(text) > 0:
        # Keep table as atomic unit
        chunk = _create_chunk(text, section_title)
        return [chunk]

    # Split by paragraphs first
    paragraphs = text.split("\n\n")

    current_chunk_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if adding this paragraph would exceed chunk_size
        if len(current_chunk_text) + len(para) + 2 <= chunk_size:
            if current_chunk_text:
                current_chunk_text += "\n\n" + para
            else:
                current_chunk_text = para
        else:
            # Save current chunk if not empty
            if current_chunk_text:
                chunk = _create_chunk(current_chunk_text, section_title)
                chunks.append(chunk)

                # Overlap: take last N chars
                if chunk_overlap > 0:
                    overlap_text = current_chunk_text[-chunk_overlap:]
                    current_chunk_text = overlap_text + "\n\n" + para
                else:
                    current_chunk_text = para
            else:
                # Paragraph itself is too large - split by sentences
                sentence_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
                for sent_chunk in sentence_chunks:
                    chunk = _create_chunk(sent_chunk, section_title)
                    chunks.append(chunk)
                current_chunk_text = ""

    # Save last chunk
    if current_chunk_text:
        chunk = _create_chunk(current_chunk_text, section_title)
        chunks.append(chunk)

    return chunks


def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text by sentences when paragraph is too large."""
    # Simple sentence split by .!?
    sentence_pattern = re.compile(r"([.!?]+\s+)")
    parts = sentence_pattern.split(text)

    sentences = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:  # End of sentence marker
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    chunks: list[str] = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Overlap
                if chunk_overlap > 0:
                    current_chunk = current_chunk[-chunk_overlap:] + sent
                else:
                    current_chunk = sent
            else:
                # Sentence itself too large - take as-is
                chunks.append(sent)
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _create_chunk(content: str, section_title: str) -> Chunk:
    """Create Chunk with auto-generated id and metadata."""
    chunk_id = hashlib.md5(content.encode()).hexdigest()

    return Chunk(
        id=chunk_id,
        content=content,
        metadata={"section_title": section_title} if section_title else {},
    )
