#!/usr/bin/env python3
"""CLI for RAG 2.0 pipeline: ingest documents, ask questions."""

import argparse
import logging
import sys

from core.config import settings


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a document into the vector store."""
    from openai import OpenAI

    from ingestion.chunker import chunk_text
    from ingestion.enricher import embed_chunks, enrich_chunks
    from ingestion.loader import load_file, load_text
    from storage.vector_store import VectorStore

    client = OpenAI(api_key=settings.openai_api_key)
    store = VectorStore()

    print(f"Initializing vector index...")
    store.init_index()

    # Load document
    file_path = args.file
    print(f"Loading: {file_path}")
    text = load_file(file_path)
    print(f"  Loaded {len(text)} characters")

    # Chunk
    print(f"Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})...")
    chunks = chunk_text(text)
    print(f"  Created {len(chunks)} chunks")

    # Enrich with contextual retrieval
    if not args.skip_enrichment:
        print("Enriching chunks with contextual retrieval...")
        chunks = enrich_chunks(chunks)
        print(f"  Enriched {len(chunks)} chunks")

    # Embed
    print("Generating embeddings...")
    chunks = embed_chunks(chunks)
    print(f"  Embedded {len(chunks)} chunks")

    # Store
    print("Storing in Neo4j vector index...")
    count = store.add_chunks(chunks)
    print(f"  Stored {count} chunks")

    total = store.count()
    print(f"\nDone! Total chunks in store: {total}")
    store.close()


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question using RAG pipeline."""
    from openai import OpenAI

    from agent.rag_agent import RAGAgent
    from retrieval.retriever import Retriever
    from storage.vector_store import VectorStore

    client = OpenAI(api_key=settings.openai_api_key)
    store = VectorStore()
    retriever = Retriever(store, client)

    if args.agent:
        # Use LangGraph agent
        agent = RAGAgent(retriever, store, client)
        print(f"Query (agent mode): {args.question}")
        result = agent.run(args.question)
    else:
        # Use simple reflect-and-answer
        from generation.reflector import reflect_and_answer

        print(f"Query: {args.question}")
        result = reflect_and_answer(args.question, retriever, client)

    print(f"\nAnswer: {result.answer}")
    print(f"\nConfidence: {result.confidence:.2f}")
    print(f"Retries: {result.retries}")

    if result.sources:
        print(f"\nSources ({len(result.sources)}):")
        for i, src in enumerate(result.sources[:5], 1):
            preview = src.chunk.content[:100].replace("\n", " ")
            print(f"  {i}. [{src.score:.3f}] {preview}...")

    store.close()


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear all chunks from vector store."""
    from storage.vector_store import VectorStore

    store = VectorStore()
    count = store.delete_all()
    print(f"Deleted {count} chunks from vector store")
    store.close()


def cmd_stats(args: argparse.Namespace) -> None:
    """Show vector store statistics."""
    from storage.vector_store import VectorStore

    store = VectorStore()
    total = store.count()
    print(f"Total chunks in store: {total}")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 2.0 Pipeline CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("file", help="Path to document file")
    p_ingest.add_argument(
        "--skip-enrichment", action="store_true",
        help="Skip contextual enrichment (faster but lower quality)"
    )

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", help="Question to ask")
    p_ask.add_argument("--agent", action="store_true", help="Use LangGraph agent mode")

    # clear
    subparsers.add_parser("clear", help="Clear all chunks")

    # stats
    subparsers.add_parser("stats", help="Show store statistics")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "clear": cmd_clear,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
