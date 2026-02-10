# RAG 2.0

Advanced Retrieval-Augmented Generation pipeline with Contextual Retrieval, Self-Reflective evaluation, and LangGraph-style agent routing.

**Benchmark: 8/10 (80%)** vs baseline 4/10 (40%) = **+40% improvement**

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              RAG 2.0 Pipeline               │
                        └─────────────────────────────────────────────┘

  INGESTION                    RETRIEVAL                    GENERATION
  ─────────                    ─────────                    ──────────
  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
  │ Docling  │──▶│Semantic │──▶│  Query   │──▶│ Vector  │──▶│  Self-   │
  │ Loader   │   │Chunker  │   │Expansion │   │ Search  │   │Reflective│
  └─────────┘   └─────────┘   │+ Multi-Q │   │(Neo4j)  │   │Evaluation│
                     │         └──────────┘   └─────────┘   └──────────┘
                     ▼                             │              │
                ┌─────────┐                   ┌─────────┐    ┌───▼────┐
                │Contextual│                  │ Cosine  │    │Retry?  │
                │Enrichment│                  │Re-rank  │    │score<3 │
                │  (LLM)   │                  └─────────┘    └───┬────┘
                └─────────┘                                      │
                     │                                      ┌────▼────┐
                     ▼                                      │Generate │
                ┌─────────┐                                 │ Answer  │
                │ OpenAI  │                                 └─────────┘
                │Embedding│
                └─────────┘
                     │
                     ▼
                ┌─────────┐
                │  Neo4j  │
                │ Vector  │
                │  Index  │
                └─────────┘
```

## Key Techniques

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Contextual Retrieval** | LLM enriches each chunk with document context before embedding | 35-49% fewer retrieval errors |
| **Query Expansion** | LLM expands short queries into detailed technical questions | Better semantic matching |
| **Multi-Query** | 3 alternative query formulations, parallel search, merged results | Broader recall |
| **Cosine Re-ranking** | Re-scores results using query-chunk cosine similarity | Precision improvement |
| **Self-Reflective RAG** | LLM evaluates relevance (1-5), retries with improved query if low | Adaptive quality |
| **Agent Routing** | Classifies query type, routes to optimal search strategy | Right tool for each query |

## Quick Start

### Prerequisites

- Python 3.12+
- Neo4j 5.x (Docker recommended)
- OpenAI API key

### 1. Setup

```bash
git clone https://github.com/vpakspace/rag-2.0.git
cd rag-2.0
pip install -r requirements.txt
```

### 2. Start Neo4j

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5
```

### 3. Configure `.env`

```env
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Models (defaults shown)
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
EMBEDDING_DIMENSIONS=1536

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
RERANK_METHOD=cosine
```

### 4. Ingest a document

```bash
python run_pipeline.py ingest data/lmcache_article.txt
```

Output:
```
Loading: data/lmcache_article.txt
  Loaded 8618 characters
Chunking (size=800, overlap=100)...
  Created 23 chunks
Enriching chunks with contextual retrieval...
  Enriched 23 chunks
Generating embeddings...
  Embedded 23 chunks
Storing in Neo4j vector index...
  Stored 23 chunks
Done! Total chunks in store: 23
```

### 5. Ask questions

```bash
# Simple mode (self-reflective RAG)
python run_pipeline.py ask "What is LMCache?"

# Agent mode (query classification + routing)
python run_pipeline.py ask "What is LMCache?" --agent
```

### 6. Run benchmark

```bash
python run_benchmark.py
```

## CLI Reference

```bash
# Ingest document (supports TXT, PDF, DOCX via Docling)
python run_pipeline.py ingest <file>
python run_pipeline.py ingest <file> --skip-enrichment  # faster, lower quality

# Ask question
python run_pipeline.py ask "<question>"
python run_pipeline.py ask "<question>" --agent  # agent mode

# Store management
python run_pipeline.py clear   # delete all chunks
python run_pipeline.py stats   # show chunk count

# Benchmark (10 LMCache questions, LLM-as-judge)
python run_benchmark.py            # reflect mode
python run_benchmark.py --agent    # agent mode
python run_benchmark.py -v         # verbose logging
```

## Benchmark Results

10 questions about LMCache article, scored by LLM-as-judge (GPT-4o-mini):

```
======================================================================
  RAG 2.0 Benchmark Results
======================================================================
  Q 1  [PASS]  conf=0.96  retries=0  What is LMCache?
  Q 2  [PASS]  conf=0.96  retries=0  What problem does LMCache solve?
  Q 3  [PASS]  conf=0.84  retries=0  What is KV cache in the context of LLMs?
  Q 4  [FAIL]  conf=0.92  retries=1  What are the three architecture patterns...
  Q 5  [PASS]  conf=0.88  retries=0  What TTFT improvement for 128K context?
  Q 6  [PASS]  conf=0.76  retries=0  What is Prefill-Decode Disaggregation?
  Q 7  [FAIL]  conf=0.68  retries=0  What are the four optimizations?
  Q 8  [PASS]  conf=0.80  retries=0  Which inference frameworks supported?
  Q 9  [PASS]  conf=0.64  retries=0  When should you NOT use LMCache?
  Q10  [PASS]  conf=0.76  retries=1  What are the tiered storage layers?
----------------------------------------------------------------------
  Accuracy: 8/10 (80%)
  Avg Confidence: 0.82
  Baseline (TKB): 4/10 (40%)
  Improvement: +40%
======================================================================
```

## Project Structure

```
rag-2.0/
├── core/
│   ├── config.py              # Pydantic settings (.env)
│   └── models.py              # Chunk, SearchResult, QAResult
├── ingestion/
│   ├── loader.py              # Docling loader (PDF/DOCX/TXT)
│   ├── chunker.py             # Semantic chunker (headers, paragraphs, tables)
│   └── enricher.py            # Contextual enrichment + embeddings
├── storage/
│   └── vector_store.py        # Neo4j Vector Index (cosine similarity)
├── retrieval/
│   ├── query_expander.py      # Query expansion + Multi-query via LLM
│   ├── reranker.py            # Cosine re-ranking (numpy)
│   └── retriever.py           # Orchestrator: expand → search → rerank
├── generation/
│   ├── generator.py           # LLM answer generation with context
│   └── reflector.py           # Self-reflective evaluation + retry
├── agent/
│   ├── rag_agent.py           # LangGraph-style agent (classify → route)
│   └── tools.py               # vector_search, focused_search, full_document_read
├── evaluation/
│   ├── benchmark.py           # LLM-as-judge benchmark runner
│   └── questions.json         # 10 LMCache test questions
├── tests/                     # 83 unit tests
├── data/
│   └── lmcache_article.txt    # Test article
├── run_pipeline.py            # CLI: ingest / ask / clear / stats
├── run_benchmark.py           # Benchmark runner
├── requirements.txt
└── .env                       # Configuration (not committed)
```

## How It Works

### Ingestion Pipeline

1. **Load** document via Docling (PDF, DOCX, HTML) or plain text reader
2. **Chunk** semantically by markdown headers, paragraphs, and sentences; tables preserved as atomic units
3. **Enrich** each chunk with LLM-generated context: _"This chunk discusses X in the context of Y"_
4. **Embed** enriched content (`context + chunk`) via OpenAI `text-embedding-3-small`
5. **Store** in Neo4j Vector Index with cosine similarity

### Retrieval Pipeline

1. **Expand** query into a detailed technical question via LLM
2. **Multi-query**: generate 3 alternative formulations
3. **Search** Neo4j vector index for each query variant (top-20 each)
4. **Merge & dedup** results by chunk ID, keeping highest scores
5. **Re-rank** using cosine similarity between query and chunk embeddings
6. **Return** top-5 results

### Generation Pipeline

1. **Evaluate** relevance of retrieved chunks (LLM scores 1-5)
2. If average score < 3.0: **retry** with LLM-generated improved query (max 2 retries)
3. **Generate** answer using best results found, with source citations
4. Return `QAResult` with answer, sources, confidence, and retry count

### Agent Mode

The LangGraph-style agent adds intelligent routing:

1. **Classify** query type: `factual` / `overview` / `comparison`
2. **Route** to optimal tool:
   - `factual` → `focused_search` (full retrieval pipeline)
   - `overview` → `full_document_read` (all chunks)
   - `comparison` → `focused_search`
3. **Evaluate** → conditional retry → **generate**

## Configuration

All settings configurable via `.env` or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | OpenAI API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | _(required)_ | Neo4j password |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI LLM model |
| `CHUNK_SIZE` | `800` | Max chunk size (chars) |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `20` | Candidates per query |
| `TOP_K_RERANK` | `5` | Final results after re-ranking |
| `MAX_RETRIES` | `2` | Max self-reflection retries |
| `RELEVANCE_THRESHOLD` | `3.0` | Min relevance score (1-5) |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Quick run
pytest tests/ -q
```

83 unit tests covering all modules with mocked OpenAI and Neo4j dependencies.

## Tech Stack

- **Python 3.12** with type hints
- **OpenAI API** (text-embedding-3-small, gpt-4o-mini)
- **Neo4j 5.x** with Vector Index (cosine similarity)
- **Docling** for document processing (PDF, DOCX, HTML)
- **NumPy** for cosine similarity computation
- **Pydantic Settings** for configuration management

## License

MIT
