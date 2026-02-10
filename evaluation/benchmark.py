"""Benchmark evaluation for RAG 2.0 pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from core.config import settings
from core.models import QAResult

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).parent / "questions.json"


def load_questions() -> list[dict]:
    """Load benchmark questions from JSON file."""
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(
    question: str,
    expected: str,
    actual: str,
    key_facts: list[str],
    openai_client: OpenAI | None = None,
) -> dict:
    """Use LLM-as-judge to evaluate answer correctness.

    Returns dict with: correct (bool), score (0 or 1), explanation.
    """
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)

    prompt = f"""Evaluate if the actual answer correctly addresses the question.
Key facts that should be present: {', '.join(key_facts)}

Question: {question}
Expected answer: {expected}
Actual answer: {actual}

Rate as CORRECT (1) or INCORRECT (0). Consider an answer correct if it captures
the key facts, even if worded differently. Minor omissions are acceptable.

Respond in this exact format:
SCORE: <0 or 1>
EXPLANATION: <one sentence explanation>"""

    response = openai_client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    text = response.choices[0].message.content or ""

    # Parse score
    score = 0
    explanation = ""
    for line in text.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                score = 0
        elif line.startswith("EXPLANATION:"):
            explanation = line.split(":", 1)[1].strip()

    return {
        "correct": score == 1,
        "score": score,
        "explanation": explanation,
    }


def run_benchmark(
    ask_fn,
    openai_client: OpenAI | None = None,
    questions: list[dict] | None = None,
) -> dict:
    """Run full benchmark and return results.

    Args:
        ask_fn: Function that takes a question string and returns QAResult
        openai_client: Optional OpenAI client for judging
        questions: Optional custom questions (defaults to questions.json)

    Returns:
        Dict with: results (list), accuracy, avg_confidence, total, correct
    """
    if questions is None:
        questions = load_questions()

    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI(api_key=settings.openai_api_key)

    results = []
    correct_count = 0
    total_confidence = 0.0

    for q in questions:
        question = q["question"]
        expected = q["expected_answer"]
        key_facts = q.get("key_facts", [])

        logger.info("Q%d: %s", q["id"], question)

        # Get answer
        qa_result = ask_fn(question)
        actual_answer = qa_result.answer if isinstance(qa_result, QAResult) else str(qa_result)

        # Judge
        judgment = judge_answer(
            question, expected, actual_answer, key_facts, openai_client
        )

        confidence = qa_result.confidence if isinstance(qa_result, QAResult) else 0.0
        retries = qa_result.retries if isinstance(qa_result, QAResult) else 0

        result = {
            "id": q["id"],
            "question": question,
            "expected": expected,
            "actual": actual_answer[:200],
            "correct": judgment["correct"],
            "score": judgment["score"],
            "explanation": judgment["explanation"],
            "confidence": confidence,
            "retries": retries,
        }
        results.append(result)

        if judgment["correct"]:
            correct_count += 1
        total_confidence += confidence

        status = "PASS" if judgment["correct"] else "FAIL"
        logger.info("  %s (confidence=%.2f, retries=%d)", status, confidence, retries)

    accuracy = correct_count / len(questions) if questions else 0.0
    avg_confidence = total_confidence / len(questions) if questions else 0.0

    return {
        "results": results,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "total": len(questions),
        "correct": correct_count,
    }


def print_benchmark_results(benchmark: dict) -> None:
    """Print formatted benchmark results."""
    print("\n" + "=" * 70)
    print(f"  RAG 2.0 Benchmark Results")
    print("=" * 70)

    for r in benchmark["results"]:
        status = "PASS" if r["correct"] else "FAIL"
        print(
            f"  Q{r['id']:2d}  [{status}]  conf={r['confidence']:.2f}  "
            f"retries={r['retries']}  {r['question'][:50]}"
        )

    print("-" * 70)
    print(
        f"  Accuracy: {benchmark['correct']}/{benchmark['total']} "
        f"({benchmark['accuracy']:.0%})"
    )
    print(f"  Avg Confidence: {benchmark['avg_confidence']:.2f}")
    print(f"  Baseline (TKB): 4/10 (40%)")
    improvement = benchmark["accuracy"] - 0.4
    print(f"  Improvement: {improvement:+.0%}")
    print("=" * 70)
