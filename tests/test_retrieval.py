"""Unit tests for retrieval pipeline components."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from core.models import Chunk, SearchResult
from retrieval.query_expander import expand_query, generate_multi_queries
from retrieval.reranker import rerank, rerank_cosine
from retrieval.retriever import Retriever


class TestQueryExpander:
    """Tests for query expansion functionality."""

    def test_expand_query_calls_openai_and_returns_string(self):
        """Test expand_query calls OpenAI and returns expanded string."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Expanded technical query"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = expand_query("short query", mock_client)

        assert result == "Expanded technical query"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"][1]["content"] == "short query"

    def test_expand_query_strips_whitespace(self):
        """Test expand_query strips leading/trailing whitespace."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="  Expanded query  \n"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = expand_query("query", mock_client)

        assert result == "Expanded query"

    def test_expand_query_handles_empty_response(self):
        """Test expand_query returns original query if OpenAI returns empty."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response

        result = expand_query("original", mock_client)

        assert result == "original"

    def test_generate_multi_queries_returns_list_of_n_strings(self):
        """Test generate_multi_queries returns list of n strings."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Query 1\nQuery 2\nQuery 3"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_multi_queries("original", n=3, openai_client=mock_client)

        assert len(result) == 3
        assert result == ["Query 1", "Query 2", "Query 3"]

    def test_generate_multi_queries_handles_fewer_than_n(self):
        """Test generate_multi_queries pads with original query if fewer than n."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Query 1"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_multi_queries("original", n=3, openai_client=mock_client)

        assert len(result) == 3
        assert result[0] == "Query 1"
        assert result[1] == "original"
        assert result[2] == "original"

    def test_generate_multi_queries_handles_more_than_n(self):
        """Test generate_multi_queries truncates if more than n returned."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Q1\nQ2\nQ3\nQ4\nQ5"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_multi_queries("original", n=3, openai_client=mock_client)

        assert len(result) == 3
        assert result == ["Q1", "Q2", "Q3"]


class TestReranker:
    """Tests for re-ranking functionality."""

    def test_rerank_cosine_sorts_by_similarity(self):
        """Test rerank_cosine correctly sorts by cosine similarity."""
        query_embedding = [1.0, 0.0, 0.0]

        # Create results with known embeddings
        chunk1 = Chunk(id="1", content="one", embedding=[1.0, 0.0, 0.0])  # cos=1.0
        chunk2 = Chunk(id="2", content="two", embedding=[0.0, 1.0, 0.0])  # cos=0.0
        chunk3 = Chunk(id="3", content="three", embedding=[0.7, 0.7, 0.0])  # cos~0.7

        results = [
            SearchResult(chunk=chunk2, score=0.5, rank=1),
            SearchResult(chunk=chunk1, score=0.3, rank=2),
            SearchResult(chunk=chunk3, score=0.8, rank=3),
        ]

        reranked = rerank_cosine(query_embedding, results, top_k=3)

        # Should be sorted: chunk1 (1.0), chunk3 (~0.7), chunk2 (0.0)
        assert len(reranked) == 3
        assert reranked[0].chunk.id == "1"
        assert reranked[0].rank == 1
        assert reranked[0].score == pytest.approx(1.0)
        assert reranked[1].chunk.id == "3"
        assert reranked[1].rank == 2
        assert reranked[2].chunk.id == "2"
        assert reranked[2].rank == 3
        assert reranked[2].score == pytest.approx(0.0)

    def test_rerank_cosine_respects_top_k(self):
        """Test rerank_cosine limits results to top_k."""
        query_embedding = [1.0, 0.0]

        chunks = [
            Chunk(id=str(i), content=f"chunk{i}", embedding=[1.0, 0.0])
            for i in range(10)
        ]
        results = [SearchResult(chunk=c, score=0.5, rank=i) for i, c in enumerate(chunks)]

        reranked = rerank_cosine(query_embedding, results, top_k=3)

        assert len(reranked) == 3
        assert all(r.rank in [1, 2, 3] for r in reranked)

    def test_rerank_cosine_handles_empty_results(self):
        """Test rerank_cosine handles empty results list."""
        query_embedding = [1.0, 0.0]
        results = []

        reranked = rerank_cosine(query_embedding, results, top_k=5)

        assert reranked == []

    def test_rerank_cosine_handles_missing_embeddings(self):
        """Test rerank_cosine handles chunks without embeddings."""
        query_embedding = [1.0, 0.0]

        chunk_with_emb = Chunk(id="1", content="has", embedding=[1.0, 0.0])
        chunk_no_emb = Chunk(id="2", content="none", embedding=[])

        results = [
            SearchResult(chunk=chunk_no_emb, score=0.5, rank=1),
            SearchResult(chunk=chunk_with_emb, score=0.3, rank=2),
        ]

        reranked = rerank_cosine(query_embedding, results, top_k=5)

        # Should only include chunk with embedding
        assert len(reranked) == 1
        assert reranked[0].chunk.id == "1"

    def test_rerank_cosine_handles_zero_norm_query(self):
        """Test rerank_cosine handles zero-norm query embedding."""
        query_embedding = [0.0, 0.0]
        chunk = Chunk(id="1", content="test", embedding=[1.0, 0.0])
        results = [SearchResult(chunk=chunk, score=0.5, rank=1)]

        reranked = rerank_cosine(query_embedding, results, top_k=5)

        # Should return original results when query norm is zero
        assert len(reranked) == 1

    def test_rerank_dispatches_to_cosine(self):
        """Test rerank dispatches to cosine method correctly."""
        query_embedding = [1.0, 0.0]
        chunk = Chunk(id="1", content="test", embedding=[1.0, 0.0])
        results = [SearchResult(chunk=chunk, score=0.5, rank=1)]

        with patch("retrieval.reranker.rerank_cosine") as mock_cosine:
            mock_cosine.return_value = results
            rerank("query", query_embedding, results, top_k=5, method="cosine")
            mock_cosine.assert_called_once_with(query_embedding, results, 5)

    def test_rerank_uses_default_settings(self):
        """Test rerank uses default top_k and method from settings."""
        query_embedding = [1.0, 0.0]
        chunk = Chunk(id="1", content="test", embedding=[1.0, 0.0])
        results = [SearchResult(chunk=chunk, score=0.5, rank=1)]

        with patch("retrieval.reranker.settings") as mock_settings:
            mock_settings.top_k_rerank = 3
            mock_settings.rerank_method = "cosine"

            with patch("retrieval.reranker.rerank_cosine") as mock_cosine:
                mock_cosine.return_value = results
                rerank("query", query_embedding, results)
                mock_cosine.assert_called_once_with(query_embedding, results, 3)


class TestRetriever:
    """Tests for Retriever orchestrator."""

    def test_get_embedding_calls_openai(self):
        """Test get_embedding calls OpenAI embeddings API."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        mock_store = MagicMock()
        retriever = Retriever(mock_store, mock_client)

        embedding = retriever.get_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch("retrieval.retriever.expand_query")
    @patch("retrieval.retriever.generate_multi_queries")
    @patch("retrieval.retriever.rerank")
    def test_retrieve_full_pipeline(
        self, mock_rerank, mock_multi_queries, mock_expand
    ):
        """Test retrieve executes full pipeline with mocks."""
        # Setup mocks
        mock_expand.return_value = "expanded query"
        mock_multi_queries.return_value = ["query1", "query2", "query3"]

        mock_client = MagicMock()
        mock_emb_response = Mock()
        mock_emb_response.data = [Mock(embedding=[1.0, 0.0])]
        mock_client.embeddings.create.return_value = mock_emb_response

        chunk = Chunk(id="1", content="test", embedding=[1.0, 0.0])
        search_result = SearchResult(chunk=chunk, score=0.9, rank=1)

        mock_store = MagicMock()
        mock_store.search.return_value = [search_result]

        mock_rerank.return_value = [search_result]

        retriever = Retriever(mock_store, mock_client)

        # Execute
        results = retriever.retrieve("original query", top_k=5)

        # Verify
        assert len(results) == 1
        mock_expand.assert_called_once_with("original query", mock_client)
        mock_multi_queries.assert_called_once_with(
            "expanded query", n=3, openai_client=mock_client
        )
        assert mock_client.embeddings.create.call_count == 4  # 1 expanded + 3 multi
        assert mock_store.search.call_count == 4
        mock_rerank.assert_called_once()

    @patch("retrieval.retriever.expand_query")
    @patch("retrieval.retriever.generate_multi_queries")
    @patch("retrieval.retriever.rerank")
    def test_retrieve_deduplication(
        self, mock_rerank, mock_multi_queries, mock_expand
    ):
        """Test retrieve deduplicates results by chunk ID."""
        mock_expand.return_value = "expanded"
        mock_multi_queries.return_value = ["q1"]

        mock_client = MagicMock()
        mock_emb_response = Mock()
        mock_emb_response.data = [Mock(embedding=[1.0, 0.0])]
        mock_client.embeddings.create.return_value = mock_emb_response

        # Same chunk ID, different scores
        chunk1 = Chunk(id="same-id", content="test", embedding=[1.0, 0.0])
        chunk2 = Chunk(id="same-id", content="test", embedding=[1.0, 0.0])

        result1 = SearchResult(chunk=chunk1, score=0.8, rank=1)
        result2 = SearchResult(chunk=chunk2, score=0.9, rank=1)  # Higher score

        mock_store = MagicMock()
        # First call returns result1, second call returns result2
        mock_store.search.side_effect = [[result1], [result2]]

        mock_rerank.return_value = [result2]

        retriever = Retriever(mock_store, mock_client)
        results = retriever.retrieve("query")

        # Should pass deduplicated results to rerank (only result2 kept)
        rerank_call_args = mock_rerank.call_args
        merged_results = rerank_call_args.kwargs["results"]
        assert len(merged_results) == 1
        assert merged_results[0].score == 0.9  # Higher score kept

    @patch("retrieval.retriever.expand_query")
    @patch("retrieval.retriever.generate_multi_queries")
    def test_retrieve_handles_empty_results(self, mock_multi_queries, mock_expand):
        """Test retrieve handles case when no results found."""
        mock_expand.return_value = "expanded"
        mock_multi_queries.return_value = ["q1"]

        mock_client = MagicMock()
        mock_emb_response = Mock()
        mock_emb_response.data = [Mock(embedding=[1.0, 0.0])]
        mock_client.embeddings.create.return_value = mock_emb_response

        mock_store = MagicMock()
        mock_store.search.return_value = []  # No results

        retriever = Retriever(mock_store, mock_client)
        results = retriever.retrieve("query")

        assert results == []

    @patch("retrieval.retriever.expand_query")
    @patch("retrieval.retriever.generate_multi_queries")
    @patch("retrieval.retriever.rerank")
    def test_retrieve_uses_settings_top_k(
        self, mock_rerank, mock_multi_queries, mock_expand
    ):
        """Test retrieve uses settings.top_k_rerank when top_k not specified."""
        mock_expand.return_value = "expanded"
        mock_multi_queries.return_value = ["q1"]

        mock_client = MagicMock()
        mock_emb_response = Mock()
        mock_emb_response.data = [Mock(embedding=[1.0, 0.0])]
        mock_client.embeddings.create.return_value = mock_emb_response

        chunk = Chunk(id="1", content="test", embedding=[1.0, 0.0])
        result = SearchResult(chunk=chunk, score=0.9, rank=1)

        mock_store = MagicMock()
        mock_store.search.return_value = [result]
        mock_rerank.return_value = [result]

        with patch("retrieval.retriever.settings") as mock_settings:
            mock_settings.top_k_rerank = 7
            mock_settings.top_k_retrieval = 20

            retriever = Retriever(mock_store, mock_client)
            retriever.retrieve("query")  # No top_k specified

            # Check rerank was called with settings.top_k_rerank
            rerank_call_args = mock_rerank.call_args
            assert rerank_call_args.kwargs["top_k"] == 7
