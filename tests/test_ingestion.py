"""Unit tests for ingestion pipeline (loader, chunker, enricher)."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from core.models import Chunk
from ingestion.chunker import chunk_text
from ingestion.enricher import embed_chunks, enrich_chunks
from ingestion.loader import load_file, load_text


class TestLoader:
    """Tests for document loader."""

    def test_load_text_passthrough(self):
        """Test load_text returns text as-is."""
        text = "Hello world"
        result = load_text(text)
        assert result == text

    def test_load_file_txt(self, tmp_path):
        """Test loading TXT file (no Docling needed)."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content\nLine 2", encoding="utf-8")

        result = load_file(str(txt_file))
        assert result == "Test content\nLine 2"

    def test_load_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_file("/nonexistent/file.txt")

    @patch("docling.document_converter.DocumentConverter")
    def test_load_file_docling(self, mock_converter_class, tmp_path):
        """Test loading PDF/DOCX via Docling."""
        # Create mock file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock DocumentConverter
        mock_result = Mock()
        mock_result.document.export_to_markdown.return_value = "# PDF Content"

        mock_converter = Mock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        result = load_file(str(pdf_file))

        assert result == "# PDF Content"
        mock_converter.convert.assert_called_once_with(str(pdf_file))


class TestChunker:
    """Tests for semantic text chunker."""

    def test_chunk_text_basic_splitting(self):
        """Test basic paragraph splitting."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.id for c in chunks)  # Auto-generated IDs

    def test_chunk_text_with_headers(self):
        """Test markdown header-based splitting."""
        text = """## Section 1

Content for section 1.

### Subsection 1.1

More content.

## Section 2

Content for section 2."""

        chunks = chunk_text(text, chunk_size=500, chunk_overlap=0)

        # Should have section_title metadata
        assert len(chunks) > 0
        section_titles = [c.metadata.get("section_title", "") for c in chunks]
        assert any("Section 1" in title for title in section_titles)

    def test_chunk_text_tables_preserved(self):
        """Test that tables (lines with |) are kept atomic."""
        table_text = """| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
| Data 3   | Data 4   |"""

        chunks = chunk_text(table_text, chunk_size=50, chunk_overlap=0)

        # Table should be kept as one chunk despite small chunk_size
        assert len(chunks) == 1
        assert "|" in chunks[0].content

    def test_chunk_text_auto_id_generation(self):
        """Test that chunk IDs are auto-generated (md5 hash)."""
        text = "Test content"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)

        assert len(chunks) == 1
        assert chunks[0].id  # ID should be non-empty
        assert len(chunks[0].id) == 32  # MD5 hash length

    def test_chunk_text_empty_input(self):
        """Test empty text returns empty list."""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=0)
        assert chunks == []

    def test_chunk_text_chunk_index(self):
        """Test chunk_index metadata is added."""
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


class TestEnricher:
    """Tests for contextual enrichment via LLM."""

    @patch("ingestion.enricher.openai.OpenAI")
    def test_enrich_chunks_calls_openai(self, mock_openai_class):
        """Test enrich_chunks calls OpenAI API."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Context sentence."))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        chunks = [Chunk(id="1", content="Test content")]

        result = enrich_chunks(chunks, document_summary="Test document")

        assert len(result) == 1
        assert result[0].context == "Context sentence."
        assert mock_client.chat.completions.create.called

    @patch("ingestion.enricher.openai.OpenAI")
    def test_enrich_chunks_generates_summary(self, mock_openai_class):
        """Test document summary generation when not provided."""
        mock_client = Mock()

        # Mock summary generation
        mock_summary_response = Mock()
        mock_summary_response.choices = [
            Mock(message=Mock(content="Summary of document"))
        ]

        # Mock context generation
        mock_context_response = Mock()
        mock_context_response.choices = [
            Mock(message=Mock(content="Context for chunk"))
        ]

        mock_client.chat.completions.create.side_effect = [
            mock_summary_response,
            mock_context_response,
        ]
        mock_openai_class.return_value = mock_client

        chunks = [Chunk(id="1", content="Test content")]

        result = enrich_chunks(chunks)  # No summary provided

        assert len(result) == 1
        assert result[0].context == "Context for chunk"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("ingestion.enricher.openai.OpenAI")
    def test_enrich_chunks_empty_input(self, mock_openai_class):
        """Test enrich_chunks with empty list."""
        result = enrich_chunks([])
        assert result == []

    @patch("ingestion.enricher.openai.OpenAI")
    def test_embed_chunks_sets_embeddings(self, mock_openai_class):
        """Test embed_chunks calls OpenAI embeddings API."""
        mock_client = Mock()

        # Mock embeddings response
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3]

        mock_response = Mock()
        mock_response.data = [mock_embedding_data]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        chunks = [Chunk(id="1", content="Test content")]

        result = embed_chunks(chunks)

        assert len(result) == 1
        assert result[0].embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch("ingestion.enricher.openai.OpenAI")
    def test_embed_chunks_empty_input(self, mock_openai_class):
        """Test embed_chunks with empty list."""
        result = embed_chunks([])
        assert result == []

    @patch("ingestion.enricher.openai.OpenAI")
    def test_embed_chunks_uses_enriched_content(self, mock_openai_class):
        """Test embed_chunks uses enriched_content property."""
        mock_client = Mock()

        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2]

        mock_response = Mock()
        mock_response.data = [mock_embedding_data]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Chunk with context (enriched_content = context + content)
        chunk = Chunk(id="1", content="Content", context="Context")

        result = embed_chunks([chunk])

        # Verify embeddings.create was called with enriched_content
        call_args = mock_client.embeddings.create.call_args
        assert "Context\n\nContent" in call_args.kwargs["input"]


class TestIntegration:
    """Integration tests for ingestion pipeline."""

    def test_full_pipeline_without_llm(self, tmp_path):
        """Test loader → chunker without LLM calls."""
        # Create test file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("## Section 1\n\nContent here.\n\n## Section 2\n\nMore content.", encoding="utf-8")

        # Load
        text = load_file(str(txt_file))
        assert "Section 1" in text

        # Chunk
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0
        assert all(c.id for c in chunks)
        assert all("section_title" in c.metadata or not c.metadata for c in chunks)
