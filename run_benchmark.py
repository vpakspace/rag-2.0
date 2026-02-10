#!/usr/bin/env python3
"""Run RAG 2.0 benchmark against LMCache test article."""

import argparse
import logging
import sys

from core.config import settings


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 2.0 Benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--agent", action="store_true", help="Use LangGraph agent mode"
    )
    args = parser.parse_args()
    setup_logging(args.verbose)

    from openai import OpenAI

    from evaluation.benchmark import print_benchmark_results, run_benchmark
    from retrieval.retriever import Retriever
    from storage.vector_store import VectorStore

    client = OpenAI(api_key=settings.openai_api_key)
    store = VectorStore()
    retriever = Retriever(store, client)

    # Build ask function based on mode
    if args.agent:
        from agent.rag_agent import RAGAgent

        agent = RAGAgent(retriever, store, client)
        print("Running benchmark in AGENT mode...")
        ask_fn = agent.run
    else:
        from generation.reflector import reflect_and_answer

        print("Running benchmark in REFLECT mode...")
        ask_fn = lambda q: reflect_and_answer(q, retriever, client)

    # Run benchmark
    results = run_benchmark(ask_fn, client)
    print_benchmark_results(results)

    store.close()

    # Exit with non-zero if below target
    if results["accuracy"] < 0.8:
        print(f"\nTarget 8/10 not met ({results['correct']}/10)")
        sys.exit(1)


if __name__ == "__main__":
    main()
