"""LangGraph RAG agent with routing: vector/focused/full-doc."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from core.config import settings
from core.models import QAResult, SearchResult

if TYPE_CHECKING:
    from openai import OpenAI

    from retrieval.retriever import Retriever
    from storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State passed through the LangGraph agent nodes."""

    query: str = ""
    query_type: str = ""  # "factual", "overview", "comparison"
    tool: str = ""  # "vector_search", "focused_search", "full_document_read"
    raw_results: list[dict] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    relevance_score: float = 0.0
    retries: int = 0
    answer: str = ""
    qa_result: QAResult | None = None


class RAGAgent:
    """LangGraph-style agent that routes queries to optimal search strategy."""

    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        openai_client: OpenAI | None = None,
    ):
        self.retriever = retriever
        self.vector_store = vector_store

        if openai_client is None:
            from openai import OpenAI

            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = openai_client

    def classify_query(self, state: AgentState) -> AgentState:
        """Classify the query type to determine search strategy."""
        response = self.openai_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the query type. Respond with ONLY one word:\n"
                        "- 'factual' for specific fact questions\n"
                        "- 'overview' for broad/summary questions\n"
                        "- 'comparison' for comparing things"
                    ),
                },
                {"role": "user", "content": state.query},
            ],
            temperature=0.0,
        )
        query_type = (response.choices[0].message.content or "factual").strip().lower()
        if query_type not in ("factual", "overview", "comparison"):
            query_type = "factual"
        state.query_type = query_type
        logger.info("Query classified as: %s", query_type)
        return state

    def select_tool(self, state: AgentState) -> AgentState:
        """Select search tool based on query type."""
        tool_map = {
            "factual": "focused_search",
            "overview": "full_document_read",
            "comparison": "focused_search",
        }
        state.tool = tool_map.get(state.query_type, "focused_search")
        logger.info("Selected tool: %s", state.tool)
        return state

    def execute_search(self, state: AgentState) -> AgentState:
        """Execute the selected search strategy."""
        from agent.tools import focused_search, full_document_read, vector_search
        from core.models import Chunk

        if state.tool == "vector_search":
            state.raw_results = vector_search(
                state.query, self.retriever, top_k=settings.top_k_rerank
            )
        elif state.tool == "full_document_read":
            doc_text = full_document_read(self.vector_store)
            state.raw_results = [
                {"id": "full_doc", "content": doc_text, "score": 1.0}
            ]
        else:  # focused_search
            state.raw_results = focused_search(
                state.query, self.retriever, top_k=settings.top_k_rerank
            )

        # Convert raw results to SearchResult objects
        state.search_results = [
            SearchResult(
                chunk=Chunk(id=r["id"], content=r["content"]),
                score=r.get("score", 0.0),
                rank=r.get("rank", i + 1),
            )
            for i, r in enumerate(state.raw_results)
        ]

        logger.info("Search returned %d results", len(state.search_results))
        return state

    def evaluate(self, state: AgentState) -> AgentState:
        """Evaluate search result relevance."""
        from generation.reflector import evaluate_relevance

        if not state.search_results:
            state.relevance_score = 0.0
            return state

        state.relevance_score = evaluate_relevance(
            state.query, state.search_results, self.openai_client
        )
        logger.info("Relevance score: %.2f", state.relevance_score)
        return state

    def should_retry(self, state: AgentState) -> bool:
        """Decide if we should retry with focused search."""
        return (
            state.relevance_score < settings.relevance_threshold
            and state.retries < settings.max_retries
            and state.tool != "focused_search"
        )

    def retry_with_focused(self, state: AgentState) -> AgentState:
        """Retry search using focused_search strategy."""
        state.tool = "focused_search"
        state.retries += 1
        logger.info("Retrying with focused_search (attempt %d)", state.retries)
        return self.execute_search(state)

    def generate(self, state: AgentState) -> AgentState:
        """Generate the final answer."""
        from generation.generator import generate_answer

        qa_result = generate_answer(
            state.query, state.search_results, self.openai_client
        )
        qa_result.retries = state.retries
        state.qa_result = qa_result
        state.answer = qa_result.answer
        logger.info("Generated answer (%d chars)", len(state.answer))
        return state

    def run(self, query: str) -> QAResult:
        """Execute the full agent pipeline.

        Flow: classify → select_tool → search → evaluate → [retry?] → generate
        """
        state = AgentState(query=query)

        # Step 1: Classify
        state = self.classify_query(state)

        # Step 2: Select tool
        state = self.select_tool(state)

        # Step 3: Execute search
        state = self.execute_search(state)

        # Step 4: Evaluate
        state = self.evaluate(state)

        # Step 5: Conditional retry
        if self.should_retry(state):
            state = self.retry_with_focused(state)
            state = self.evaluate(state)

        # Step 6: Generate answer
        state = self.generate(state)

        return state.qa_result
