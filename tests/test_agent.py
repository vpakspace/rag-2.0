"""Unit tests for RAG agent and tools."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from agent.rag_agent import AgentState, RAGAgent
from agent.tools import focused_search, full_document_read, vector_search
from core.models import Chunk, QAResult, SearchResult


class TestAgentTools:
    def test_vector_search(self):
        mock_retriever = MagicMock()
        mock_retriever.get_embedding.return_value = [0.1, 0.2]
        mock_retriever.vector_store.search.return_value = [
            SearchResult(
                chunk=Chunk(id="c1", content="Test", context="Ctx"),
                score=0.9,
                rank=1,
            )
        ]

        results = vector_search("query", mock_retriever, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "c1"
        assert results[0]["score"] == 0.9
        mock_retriever.get_embedding.assert_called_once_with("query")

    def test_focused_search(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            SearchResult(
                chunk=Chunk(id="c1", content="Found", context="Ctx"),
                score=0.95,
                rank=1,
            )
        ]

        results = focused_search("deep query", mock_retriever, top_k=3)

        assert len(results) == 1
        assert results[0]["rank"] == 1
        mock_retriever.retrieve.assert_called_once_with("deep query", top_k=3)

    def test_full_document_read(self):
        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_store._driver.session.return_value.__enter__ = Mock(
            return_value=mock_session
        )
        mock_store._driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_session.run.return_value = [
            {"content": "Part 1", "metadata": "{}"},
            {"content": "Part 2", "metadata": "{}"},
        ]

        result = full_document_read(mock_store)

        assert "Part 1" in result
        assert "Part 2" in result


class TestRAGAgent:
    @pytest.fixture
    def mock_openai(self):
        client = MagicMock()
        response = Mock()
        response.choices = [Mock(message=Mock(content="factual"))]
        client.chat.completions.create.return_value = response
        return client

    @pytest.fixture
    def agent(self, mock_openai):
        mock_retriever = MagicMock()
        mock_store = MagicMock()
        return RAGAgent(mock_retriever, mock_store, mock_openai)

    def test_classify_query_factual(self, agent, mock_openai):
        mock_openai.chat.completions.create.return_value.choices[
            0
        ].message.content = "factual"
        state = AgentState(query="What is X?")

        state = agent.classify_query(state)

        assert state.query_type == "factual"

    def test_classify_query_overview(self, agent, mock_openai):
        mock_openai.chat.completions.create.return_value.choices[
            0
        ].message.content = "overview"
        state = AgentState(query="Summarize the document")

        state = agent.classify_query(state)

        assert state.query_type == "overview"

    def test_classify_query_invalid_defaults_to_factual(self, agent, mock_openai):
        mock_openai.chat.completions.create.return_value.choices[
            0
        ].message.content = "unknown_type"
        state = AgentState(query="test")

        state = agent.classify_query(state)

        assert state.query_type == "factual"

    def test_select_tool_factual(self, agent):
        state = AgentState(query_type="factual")
        state = agent.select_tool(state)
        assert state.tool == "focused_search"

    def test_select_tool_overview(self, agent):
        state = AgentState(query_type="overview")
        state = agent.select_tool(state)
        assert state.tool == "full_document_read"

    def test_should_retry_low_relevance(self, agent):
        state = AgentState(
            relevance_score=1.0, retries=0, tool="vector_search"
        )
        assert agent.should_retry(state) is True

    def test_should_not_retry_high_relevance(self, agent):
        state = AgentState(
            relevance_score=4.0, retries=0, tool="vector_search"
        )
        assert agent.should_retry(state) is False

    def test_should_not_retry_already_focused(self, agent):
        state = AgentState(
            relevance_score=1.0, retries=0, tool="focused_search"
        )
        assert agent.should_retry(state) is False

    def test_should_not_retry_max_retries(self, agent):
        state = AgentState(
            relevance_score=1.0, retries=2, tool="vector_search"
        )
        assert agent.should_retry(state) is False

    @patch("agent.rag_agent.RAGAgent.execute_search")
    @patch("agent.rag_agent.RAGAgent.evaluate")
    @patch("agent.rag_agent.RAGAgent.generate")
    def test_run_full_pipeline(
        self, mock_generate, mock_evaluate, mock_execute, agent, mock_openai
    ):
        # Setup
        qa_result = QAResult(
            answer="Test answer", confidence=0.8, query="What is X?"
        )

        def set_results(state):
            state.search_results = [
                SearchResult(
                    chunk=Chunk(id="c1", content="Test"), score=0.9, rank=1
                )
            ]
            return state

        def set_relevance(state):
            state.relevance_score = 4.0
            return state

        def set_answer(state):
            state.qa_result = qa_result
            state.answer = qa_result.answer
            return state

        mock_execute.side_effect = set_results
        mock_evaluate.side_effect = set_relevance
        mock_generate.side_effect = set_answer

        result = agent.run("What is X?")

        assert result is not None
        assert result.answer == "Test answer"
        mock_execute.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_generate.assert_called_once()


class TestBenchmark:
    def test_load_questions(self):
        from evaluation.benchmark import load_questions

        questions = load_questions()
        assert len(questions) == 10
        assert questions[0]["id"] == 1
        assert "question" in questions[0]
        assert "expected_answer" in questions[0]
        assert "key_facts" in questions[0]

    def test_judge_answer_correct(self):
        from evaluation.benchmark import judge_answer

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "SCORE: 1\nEXPLANATION: Correct answer."

        result = judge_answer(
            "What is X?", "X is Y", "X is Y indeed", ["X", "Y"], mock_client
        )

        assert result["correct"] is True
        assert result["score"] == 1

    def test_judge_answer_incorrect(self):
        from evaluation.benchmark import judge_answer

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "SCORE: 0\nEXPLANATION: Missing key facts."

        result = judge_answer(
            "What is X?", "X is Y", "I don't know", ["X", "Y"], mock_client
        )

        assert result["correct"] is False
        assert result["score"] == 0

    def test_run_benchmark(self):
        from evaluation.benchmark import run_benchmark

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[
            0
        ].message.content = "SCORE: 1\nEXPLANATION: Correct."

        questions = [
            {
                "id": 1,
                "question": "What?",
                "expected_answer": "Answer",
                "key_facts": ["fact"],
            }
        ]

        mock_qa = QAResult(answer="Answer", confidence=0.8)
        ask_fn = lambda q: mock_qa

        results = run_benchmark(ask_fn, mock_client, questions)

        assert results["total"] == 1
        assert results["correct"] == 1
        assert results["accuracy"] == 1.0
