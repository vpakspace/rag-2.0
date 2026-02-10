"""Unit tests for generation and reflection modules."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from core.models import Chunk, QAResult, SearchResult
from generation.generator import generate_answer
from generation.reflector import (
    evaluate_relevance,
    generate_retry_query,
    reflect_and_answer,
)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk1",
            content="Python is a high-level programming language.",
            context="Programming Languages",
        ),
        Chunk(
            id="chunk2",
            content="Python was created by Guido van Rossum.",
            context="History",
        ),
        Chunk(
            id="chunk3",
            content="Python supports object-oriented programming.",
            context="Features",
        ),
    ]


@pytest.fixture
def sample_results(sample_chunks):
    """Create sample search results."""
    return [
        SearchResult(chunk=sample_chunks[0], score=0.9, rank=1),
        SearchResult(chunk=sample_chunks[1], score=0.8, rank=2),
        SearchResult(chunk=sample_chunks[2], score=0.7, rank=3),
    ]


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Test answer based on chunks 1 and 2."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_generate_answer_creates_correct_prompt(sample_results, mock_openai_client):
    """Test that generate_answer creates correct prompt with context chunks."""
    query = "What is Python?"

    qa_result = generate_answer(query, sample_results, mock_openai_client)

    # Verify OpenAI was called
    assert mock_openai_client.chat.completions.create.called

    # Get the call arguments
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    # Check system message
    assert messages[0]["role"] == "system"
    assert "precise Q&A assistant" in messages[0]["content"]
    assert "ONLY based on the provided context" in messages[0]["content"]

    # Check user message contains query and chunks
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert query in user_content
    assert "[Chunk 1]" in user_content
    assert "[Chunk 2]" in user_content
    assert "[Chunk 3]" in user_content
    assert "Python is a high-level programming language" in user_content


def test_generate_answer_returns_qa_result(sample_results, mock_openai_client):
    """Test that generate_answer returns QAResult with answer and sources."""
    query = "What is Python?"

    qa_result = generate_answer(query, sample_results, mock_openai_client)

    # Check result type and fields
    assert isinstance(qa_result, QAResult)
    assert qa_result.answer == "Test answer based on chunks 1 and 2."
    assert qa_result.sources == sample_results
    assert qa_result.confidence == 0.5  # Default confidence
    assert qa_result.query == query


def test_generate_answer_handles_empty_results(mock_openai_client):
    """Test that generate_answer handles empty results gracefully."""
    query = "What is Python?"

    qa_result = generate_answer(query, [], mock_openai_client)

    # OpenAI should not be called for empty results
    assert not mock_openai_client.chat.completions.create.called

    # Check result
    assert isinstance(qa_result, QAResult)
    assert "don't have enough context" in qa_result.answer
    assert qa_result.sources == []
    assert qa_result.confidence == 0.0


def test_evaluate_relevance_returns_float_score(sample_results):
    """Test that evaluate_relevance returns float score."""
    query = "What is Python?"

    # Mock OpenAI to return scores
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "4, 5, 3"  # Scores for 3 chunks
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    score = evaluate_relevance(query, sample_results, mock_client)

    # Check score is float and calculated correctly
    assert isinstance(score, float)
    assert score == pytest.approx((4 + 5 + 3) / 3, rel=0.01)


def test_evaluate_relevance_handles_malformed_response(sample_results):
    """Test that evaluate_relevance handles malformed LLM response (non-numeric)."""
    query = "What is Python?"

    # Mock OpenAI to return malformed response
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "not, numbers, here"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    score = evaluate_relevance(query, sample_results, mock_client)

    # Should return default score for unparseable values
    assert isinstance(score, float)
    assert score == 2.5  # Default middle score


def test_evaluate_relevance_handles_empty_results():
    """Test that evaluate_relevance handles empty results."""
    query = "What is Python?"
    mock_client = Mock()

    score = evaluate_relevance(query, [], mock_client)

    # Should not call OpenAI for empty results
    assert not mock_client.chat.completions.create.called
    assert score == 0.0


def test_generate_retry_query_returns_new_query(sample_results):
    """Test that generate_retry_query returns new query string."""
    query = "What is Python?"

    # Mock OpenAI to return improved query
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "What is Python programming language and its features?"
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    retry_query = generate_retry_query(query, sample_results, mock_client)

    # Check that new query is returned
    assert isinstance(retry_query, str)
    assert retry_query == "What is Python programming language and its features?"
    assert retry_query != query


def test_reflect_and_answer_no_retry_high_relevance(sample_results):
    """Test reflect_and_answer with high relevance (no retry)."""
    query = "What is Python?"

    # Mock retriever
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = sample_results

    # Mock OpenAI client
    mock_client = Mock()

    # First call: evaluate_relevance -> high score (4.0)
    # Second call: generate_answer -> answer
    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        if call_count[0] == 1:
            # First call: evaluate_relevance
            mock_message.content = "4, 5, 4"  # Avg 4.33 > 3.0 threshold
        else:
            # Second call: generate_answer
            mock_message.content = "Python is a programming language."

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.completions.create = mock_create

    result = reflect_and_answer(query, mock_retriever, mock_client)

    # Should retrieve only once (no retry)
    assert mock_retriever.retrieve.call_count == 1

    # Check result
    assert isinstance(result, QAResult)
    assert result.retries == 0
    assert "Python is a programming language" in result.answer


def test_reflect_and_answer_triggers_retry_low_relevance(sample_results):
    """Test reflect_and_answer with low relevance triggers retry."""
    query = "What is Python?"

    # Mock retriever
    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = sample_results

    # Mock OpenAI client
    mock_client = Mock()

    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        if call_count[0] == 1:
            # First evaluate: low score
            mock_message.content = "2, 2, 2"  # Avg 2.0 < 3.0 threshold
        elif call_count[0] == 2:
            # Generate retry query
            mock_message.content = "What is Python programming language?"
        elif call_count[0] == 3:
            # Second evaluate: high score
            mock_message.content = "4, 4, 4"  # Avg 4.0 > 3.0 threshold
        else:
            # Generate answer
            mock_message.content = "Python is a high-level language."

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.completions.create = mock_create

    result = reflect_and_answer(query, mock_retriever, mock_client)

    # Should retrieve twice (1 initial + 1 retry)
    assert mock_retriever.retrieve.call_count == 2

    # Check result
    assert isinstance(result, QAResult)
    assert result.retries == 1
    assert result.expanded_query != ""


def test_reflect_and_answer_max_retries_limit():
    """Test reflect_and_answer respects max retries limit."""
    query = "What is Python?"

    # Mock retriever
    mock_retriever = Mock()
    sample_chunk = Chunk(id="c1", content="Test content")
    sample_results = [SearchResult(chunk=sample_chunk, score=0.5, rank=1)]
    mock_retriever.retrieve.return_value = sample_results

    # Mock OpenAI client to always return low scores
    mock_client = Mock()

    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Always return low relevance scores or retry queries
        if "Rate the relevance" in kwargs["messages"][0]["content"]:
            mock_message.content = "1"  # Low score
        elif "better search query" in kwargs["messages"][0]["content"]:
            mock_message.content = f"Retry query {call_count[0]}"
        else:
            mock_message.content = "Final answer."

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.completions.create = mock_create

    result = reflect_and_answer(query, mock_retriever, mock_client)

    # Should retrieve max_retries + 1 times (1 initial + 2 retries = 3 total)
    assert mock_retriever.retrieve.call_count == 3

    # Check result
    assert isinstance(result, QAResult)
    assert result.retries == 2  # Max retries


def test_reflect_and_answer_uses_best_results():
    """Test reflect_and_answer uses best results after retries exhausted."""
    query = "What is Python?"

    # Mock retriever to return different results
    mock_retriever = Mock()

    results_batch1 = [SearchResult(chunk=Chunk(id="c1", content="Batch 1"), score=0.5, rank=1)]
    results_batch2 = [SearchResult(chunk=Chunk(id="c2", content="Batch 2"), score=0.6, rank=1)]
    results_batch3 = [SearchResult(chunk=Chunk(id="c3", content="Batch 3"), score=0.4, rank=1)]

    mock_retriever.retrieve.side_effect = [results_batch1, results_batch2, results_batch3]

    # Mock OpenAI client
    mock_client = Mock()

    call_count = [0]

    def mock_create(**kwargs):
        call_count[0] += 1
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Always return low relevance to trigger all retries
        if "Rate the relevance" in str(kwargs.get("messages", [])):
            # Return different scores for different batches
            if call_count[0] == 1:
                mock_message.content = "2"  # Batch 1: score 2.0
            elif call_count[0] == 3:
                mock_message.content = "3.5"  # Batch 2: score 3.5 (best)
            else:
                mock_message.content = "1.5"  # Batch 3: score 1.5
        elif "better search query" in str(kwargs.get("messages", [])):
            mock_message.content = f"Retry {call_count[0]}"
        else:
            mock_message.content = "Answer based on best results."

        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        return mock_response

    mock_client.chat.completions.create = mock_create

    with patch("generation.reflector.settings") as mock_settings:
        mock_settings.max_retries = 2
        mock_settings.relevance_threshold = 5.0  # Set very high to trigger all retries
        mock_settings.llm_model = "gpt-4o-mini"

        result = reflect_and_answer(query, mock_retriever, mock_client)

    # Should use batch 2 results (highest score 3.5)
    assert isinstance(result, QAResult)
    assert result.retries == 2
    assert len(result.sources) > 0
    # Confidence should be normalized score of best batch (3.5/5.0 = 0.7)
    assert result.confidence == pytest.approx(0.7, abs=0.05)
