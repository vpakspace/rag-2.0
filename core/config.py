"""RAG 2.0 configuration via Pydantic settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    embedding_dimensions: int = 1536

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "temporal_kb_2026"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    rerank_method: str = "cosine"  # "cosine" or "cross-encoder"

    # Reflection
    max_retries: int = 2
    relevance_threshold: float = 3.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
