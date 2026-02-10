# RAG 2.0 — Project Memory

## Overview

Advanced RAG pipeline: Contextual Retrieval + Self-Reflective RAG + Agentic RAG.

**GitHub**: https://github.com/vpakspace/rag-2.0
**Stack**: Python 3.12, Neo4j 5.x (Vector Index), OpenAI (text-embedding-3-small + gpt-4o-mini), Docling, Streamlit
**Tests**: 83 unit tests

## Benchmark Results

| Mode | Accuracy | Improvement vs Baseline |
|------|----------|------------------------|
| Baseline (TKB) | 4/10 (40%) | — |
| Reflect mode | 8/10 (80%) | +40% |
| **Agent mode** | **9/10 (90%)** | **+50%** |

## Architecture

### Ingestion Pipeline
`load_file()` → `chunk_text()` → `enrich_chunks()` → `embed_chunks()` → `store.add_chunks()`

### Retrieval Pipeline
`expand_query()` → `generate_multi_queries()` → `vector_store.search()` → `rerank()` → top-5

### Generation Pipeline
`reflect_and_answer()`: retrieve → evaluate relevance (1-5) → retry if < 3.0 (max 2) → generate answer

### Agent Mode (LangGraph-style)
`RAGAgent.run()`: classify query (factual/overview/comparison) → route to tool → evaluate → generate

## Project Structure

```
rag-2.0/
├── core/
│   ├── config.py              # Pydantic settings (.env)
│   └── models.py              # Chunk, SearchResult, QAResult
├── ingestion/
│   ├── loader.py              # Docling loader (PDF/DOCX/TXT) + GPU toggle
│   ├── chunker.py             # Semantic chunker (headers, paragraphs, tables)
│   └── enricher.py            # Contextual enrichment + embeddings
├── storage/
│   └── vector_store.py        # Neo4j Vector Index (cosine similarity)
├── retrieval/
│   ├── query_expander.py      # Query expansion + Multi-query
│   ├── reranker.py            # Cosine re-ranking
│   └── retriever.py           # Orchestrator: expand → search → rerank
├── generation/
│   ├── generator.py           # LLM answer generation
│   └── reflector.py           # Self-reflective evaluation + retry
├── agent/
│   ├── rag_agent.py           # LangGraph-style agent (classify → route)
│   └── tools.py               # vector_search, focused_search, full_document_read
├── evaluation/
│   ├── benchmark.py           # LLM-as-judge benchmark runner
│   └── questions.json         # 10 LMCache test questions
├── ui/
│   ├── __init__.py
│   └── i18n.py                # ~50 ключей перевода RU/EN
├── tests/                     # 83 unit tests
├── data/
│   └── lmcache_article.txt    # Test article
├── streamlit_app.py           # Streamlit UI (4 tabs, port 8502)
├── run_streamlit.sh           # Скрипт запуска UI
├── run_pipeline.py            # CLI: ingest / ask / clear / stats
├── run_benchmark.py           # Benchmark runner
└── requirements.txt
```

## Key API Signatures

```python
# Ingestion
load_file(file_path: str, use_gpu: bool = False) -> str
chunk_text(text: str, chunk_size=None, chunk_overlap=None) -> list[Chunk]
enrich_chunks(chunks: list[Chunk], document_summary: str = "") -> list[Chunk]  # creates OpenAI client internally
embed_chunks(chunks: list[Chunk]) -> list[Chunk]  # creates OpenAI client internally

# Storage
VectorStore(driver=None)  # creates Neo4j driver from settings if None
store.init_index()
store.add_chunks(chunks) -> int
store.search(query_embedding, top_k=None) -> list[SearchResult]
store.count() -> int
store.delete_all() -> int
store.close()

# Retrieval
Retriever(vector_store, openai_client=None)
retriever.retrieve(query, top_k=None) -> list[SearchResult]

# Generation
reflect_and_answer(query, retriever, openai_client=None) -> QAResult
generate_answer(query, results, openai_client=None) -> QAResult

# Agent
RAGAgent(retriever, vector_store, openai_client=None)
agent.run(query) -> QAResult

# Benchmark
run_benchmark(ask_fn, openai_client=None, questions=None) -> dict
load_questions() -> list[dict]
judge_answer(question, expected, actual, key_facts, openai_client=None) -> dict
```

## Important: Function Internals

- `enrich_chunks()` и `embed_chunks()` создают OpenAI client **внутри** (не принимают его как параметр)
- `reflect_and_answer()`, `RAGAgent`, `Retriever` — принимают `openai_client` как опциональный параметр
- `VectorStore` создаёт Neo4j driver из `settings` если не передан

## Streamlit UI

**Порт**: 8502 (не конфликтует с temporal-knowledge-base на 8501)
**Запуск**: `./run_streamlit.sh` или `streamlit run streamlit_app.py --server.port 8502`

### 4 вкладки
1. **Ingest** — загрузка документов (TXT/PDF/DOCX/PPTX/XLSX/HTML), progress bar, skip enrichment
2. **Search & Q&A** — вопрос-ответ, toggle simple/agent mode, confidence bar, sources expander
3. **Benchmark** — 10 вопросов, progress bar, таблица PASS/FAIL, accuracy metrics
4. **Settings** — конфигурация (read-only), count чанков, clear DB (DELETE), re-init

### Sidebar
- Language selector (EN/RU) с session_state persistence
- GPU Acceleration toggle для Docling PDF pipeline

### i18n
- `ui/i18n.py`: `get_translator(lang)` → closure `t(key, **kwargs)`
- ~50 ключей перевода, паттерн из temporal-knowledge-base

## GPU Acceleration (Docling)

- `load_file(file_path, use_gpu=True)` включает `AcceleratorDevice.AUTO` для PDF
- Import: `from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions`
- Только для PDF файлов (другие форматы не поддерживают GPU pipeline)
- RTX 4080 Laptop GPU — существенное ускорение (подтверждено пользователем)

## Configuration

Все через `.env` или environment variables (Pydantic Settings):
- `OPENAI_API_KEY` — обязательно
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `EMBEDDING_MODEL`, `LLM_MODEL`, `EMBEDDING_DIMENSIONS`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `TOP_K_RETRIEVAL`, `TOP_K_RERANK`, `RERANK_METHOD`
- `MAX_RETRIES`, `RELEVANCE_THRESHOLD`

## Commits

| Commit | Description |
|--------|-------------|
| `9d4d918` | feat: RAG 2.0 pipeline — 8/10 benchmark |
| `e8304c9` | docs: README with architecture and benchmarks |
| `1671f2f` | feat: Streamlit UI (4 tabs, i18n, GPU toggle), agent mode 9/10 |

## Development Notes

- Запуск тестов: `cd ~/rag-2.0 && python -m pytest tests/ -x -q`
- Neo4j: использует тот же контейнер `temporal-kb-neo4j` (порты 7474/7687)
- `@st.cache_resource` для VectorStore, OpenAI client, Retriever — создаются один раз
- Файлы загружаются через `tempfile.NamedTemporaryFile` → `load_file(tmp_path)`
