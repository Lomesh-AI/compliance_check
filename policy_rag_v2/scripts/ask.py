"""
ask.py — Interactive DPDP Act QA with the full Phase 2 pipeline.

Fixes vs previous version:
  - Queries and answers now logged to data/eval/query_log.jsonl (#17 from PDF feedback)
  - Context token count estimated and warned if it approaches LLM limits (#9 from PDF feedback)

Pipeline:
  query → RRF fusion → cross-encoder rerank → LLM → NLI faithfulness check → answer

Usage:
    python scripts/ask.py
    python scripts/ask.py --query "What are the rights of a Data Principal?"
    python scripts/ask.py --no-guard   # skip NLI check (faster)
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import UnifiedRetriever
from src.reranker.reranker import CrossEncoderReranker, FaithfulnessGuard
from src.generation.generator import ComplianceGenerator
from src.utils.config import DENSE_TOP_K, RERANK_TOP_K, EVAL_DIR

logging.basicConfig(level=logging.WARNING)

# Approximate token budget warning threshold.
# Groq llama3-8b-8192 context = 8192 tokens.
# We warn if estimated context exceeds 6000 tokens (leaves 2000 for answer).
# 1 token ≈ 0.75 words (rough heuristic, no tiktoken needed)
TOKEN_WARN_THRESHOLD = 6000
WORDS_PER_TOKEN = 0.75


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) / WORDS_PER_TOKEN)


def _log_query(log_path: Path, query: str, answer: str, faithfulness: Optional[float]):
    """Append query + answer to a JSONL log file for auditing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp":   datetime.utcnow().isoformat(),
        "query":       query,
        "answer":      answer,
        "faithfulness": faithfulness,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def answer(
    query: str,
    retriever: UnifiedRetriever,
    reranker: CrossEncoderReranker,
    generator: ComplianceGenerator,
    guard: Optional[FaithfulnessGuard],
    log_path: Optional[Path] = None,
) -> None:
    print(f"\nQuery: {query}")
    print("─" * 60)

    # Retrieve + rerank
    rrf_res    = retriever.rrf_search(query, top_k=DENSE_TOP_K)
    top_chunks = reranker.rerank(query, rrf_res, top_k=RERANK_TOP_K)

    print(f"Retrieved {len(rrf_res)} candidates → reranked to top {len(top_chunks)}")
    for c in top_chunks:
        print(f"  [chunk_{c['id']}] {c['filename']}  score={c.get('rerank_score', 0):.3f}")

    # Context token estimate warning
    context_words = sum(len(c["text"].split()) for c in top_chunks)
    est_tokens    = _estimate_tokens(" ".join(c["text"] for c in top_chunks))
    if est_tokens > TOKEN_WARN_THRESHOLD:
        print(f"\n⚠ Context is ~{est_tokens} tokens "
              f"(>{TOKEN_WARN_THRESHOLD} threshold). "
              "May exceed LLM context window. Consider reducing RERANK_TOP_K in config.py.")

    # Generate
    raw_answer    = generator.answer_query(query, top_chunks)
    faithfulness  = None

    if guard:
        faith = guard.check(raw_answer, top_chunks)
        faithfulness = faith["faithfulness"]
        print(f"\nFaithfulness: {faith['faithfulness']:.2%} "
              f"({faith['supported_count']}/{faith['total_count']} sentences grounded)")
        if faith["flagged_sentences"]:
            print(f"⚠ {len(faith['flagged_sentences'])} sentence(s) not grounded:")
            for fs in faith["flagged_sentences"]:
                print(f"  [{fs['entailment_score']:.2f}] {fs['sentence'][:100]}...")
        final_answer = faith["filtered_answer"]
        print(f"\nAnswer (grounded):\n{final_answer}")
    else:
        final_answer = raw_answer
        print(f"\nAnswer:\n{final_answer}")

    print("─" * 60)

    # Log query + answer
    if log_path:
        _log_query(log_path, query, final_answer, faithfulness)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query",    type=str, default=None, help="Single query (non-interactive)")
    ap.add_argument("--no-guard", action="store_true",    help="Disable NLI faithfulness guard")
    ap.add_argument("--no-log",   action="store_true",    help="Disable query logging to file")
    args = ap.parse_args()

    log_path = None if args.no_log else (EVAL_DIR / "query_log.jsonl")

    print("Loading retriever...")
    retriever = UnifiedRetriever()
    print("Loading reranker...")
    reranker  = CrossEncoderReranker()

    guard = None
    if not args.no_guard:
        print("Loading NLI guard...")
        guard = FaithfulnessGuard()

    print("Loading generator...")
    generator = ComplianceGenerator()
    print("Ready.\n")

    if log_path:
        print(f"Query log: {log_path}\n")

    if args.query:
        answer(args.query, retriever, reranker, generator, guard, log_path)
        return

    print("Ask questions about the DPDP Act. Type 'exit' to quit.\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            break
        answer(q, retriever, reranker, generator, guard, log_path)


if __name__ == "__main__":
    main()
