"""
run_eval.py — Full evaluation harness with ablation, citation verification,
compliance gold eval, and unverified-label warnings.

Sections:
  1. RETRIEVAL — Recall@k, MRR, nDCG@k for 4 modes:
       dense-only | sparse-only | rrf (no rerank) | rrf+rerank
     Warns if gold entries have manual_verified: false.

  2. GENERATION + HALLUCINATION
       Phase 1: RRF → LLM (no guard)
       Phase 2: RRF → rerank → LLM → NLI guard
       Metric: faithfulness delta (Phase 2 - Phase 1)
       + CITATION VERIFICATION: checks cited chunk IDs actually exist in context

  3. COMPLIANCE GOLD
       Runs check_compliance_for_rule() on manually authored policy excerpts
       in gold_compliance.json and compares predicted vs expected verdicts.
       Reports accuracy, precision, recall per verdict class.

Usage:
    python src/eval/run_eval.py
    python src/eval/run_eval.py --skip-generation
    python src/eval/run_eval.py --skip-compliance
    python src/eval/run_eval.py --out data/eval/report.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Dict

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    CHUNKS_PATH, FAISS_PATH, KG_TRIPLES,
    GOLD_PATH, GOLD_GEN_PATH, EVAL_DIR, DOCS_DIR,
    RERANK_TOP_K, DENSE_TOP_K,
)
from src.retrieval.retriever import UnifiedRetriever
from src.reranker.reranker import CrossEncoderReranker, FaithfulnessGuard
from src.generation.generator import ComplianceGenerator

log = logging.getLogger("run_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GOLD_COMP_PATH = EVAL_DIR / "gold_compliance.json"


# ── Metric helpers ────────────────────────────────────────────────────────────

def recall_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(relevant) & set(retrieved[:k])) / len(set(relevant))

def mrr(relevant: List[int], retrieved: List[int]) -> float:
    for i, rid in enumerate(retrieved):
        if rid in set(relevant):
            return 1.0 / (i + 1)
    return 0.0

def dcg_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
    score = 0.0
    for i, rid in enumerate(retrieved[:k]):
        if rid in set(relevant):
            score += 1.0 / np.log2(i + 2)
    return score

def ndcg_at_k(relevant: List[int], retrieved: List[int], k: int) -> float:
    dcg   = dcg_at_k(relevant, retrieved, k)
    ideal = dcg_at_k(relevant, relevant, k)
    return dcg / ideal if ideal > 0 else 0.0


# ── Citation verification ─────────────────────────────────────────────────────

def verify_citations(answer: str, context_chunks: List[Dict]) -> Dict:
    """
    Check that every [chunk_N] citation in the answer refers to a chunk ID
    that was actually in the context. The LLM can hallucinate chunk IDs.

    Returns:
        valid_citations   : list of int IDs that exist in context
        invalid_citations : list of int IDs not in context (hallucinated)
        citation_accuracy : valid / total (0.0–1.0)
    """
    cited_ids = [int(m) for m in re.findall(r'\[chunk_(\d+)\]', answer)]
    if not cited_ids:
        return {
            "valid_citations":    [],
            "invalid_citations":  [],
            "citation_accuracy":  None,   # None = no citations present
            "citations_present":  False,
        }

    context_ids = {c.get("id") for c in context_chunks}
    valid   = [cid for cid in cited_ids if cid in context_ids]
    invalid = [cid for cid in cited_ids if cid not in context_ids]

    return {
        "valid_citations":    valid,
        "invalid_citations":  invalid,
        "citation_accuracy":  len(valid) / len(cited_ids) if cited_ids else 0.0,
        "citations_present":  True,
    }


# ── Section 1: Retrieval with ablation ───────────────────────────────────────

def eval_retrieval(retriever: UnifiedRetriever, reranker: CrossEncoderReranker) -> Dict:
    if not GOLD_PATH.exists():
        log.error("Gold retrieval file not found: %s — run build_gold.py first.", GOLD_PATH)
        return {}

    with open(GOLD_PATH) as f:
        gold = json.load(f)

    # Warn about unverified labels
    unverified = [g for g in gold if not g.get("manual_verified", False)]
    if unverified:
        log.warning(
            "⚠ %d / %d retrieval gold entries have manual_verified=false. "
            "Metrics may be inflated due to circular labeling. "
            "Open data/eval/gold_retrieval.json and set manual_verified=true "
            "after checking each entry manually.",
            len(unverified), len(gold),
        )

    K_VALUES = [1, 3, 5, 10]
    # ABLATION: 4 modes to prove each component adds value
    methods   = ["dense", "sparse", "rrf", "rrf_rerank"]
    scores    = {m: {f"recall@{k}": [] for k in K_VALUES} for m in methods}
    for m in methods:
        scores[m]["mrr"]     = []
        scores[m]["ndcg@10"] = []

    usable = 0
    for item in gold:
        query    = item["query"]
        relevant = item["relevant_chunk_ids"]
        if not relevant:
            log.warning("Skipping query with no relevant IDs: %s", query[:50])
            continue
        usable += 1

        dense_res  = retriever.dense_search(query, k=DENSE_TOP_K)
        sparse_res = retriever.sparse_search(query, k=DENSE_TOP_K)
        rrf_res    = retriever.rrf_search(query, top_k=DENSE_TOP_K)
        rrf_rerank = reranker.rerank(query, rrf_res, top_k=RERANK_TOP_K)

        results_map = {
            "dense":      [r["id"] for r in dense_res],
            "sparse":     [r["id"] for r in sparse_res],
            "rrf":        [r["id"] for r in rrf_res],
            "rrf_rerank": [r["id"] for r in rrf_rerank],
        }

        for m, ids in results_map.items():
            for k in K_VALUES:
                scores[m][f"recall@{k}"].append(recall_at_k(relevant, ids, k))
            scores[m]["mrr"].append(mrr(relevant, ids))
            scores[m]["ndcg@10"].append(ndcg_at_k(relevant, ids, 10))

    averaged = {}
    for m in methods:
        averaged[m] = {metric: round(float(np.mean(vals)), 4)
                       for metric, vals in scores[m].items() if vals}

    return {
        "n_queries":      len(gold),
        "usable_queries": usable,
        "unverified":     len(unverified),
        "methods":        averaged,
    }


# ── Section 2: Generation + hallucination + citation verification ─────────────

def eval_generation(
    retriever:  UnifiedRetriever,
    reranker:   CrossEncoderReranker,
    guard:      FaithfulnessGuard,
    generator:  ComplianceGenerator,
) -> Dict:
    if not GOLD_GEN_PATH.exists():
        log.error("Gold generation file not found: %s — run build_gold.py first.", GOLD_GEN_PATH)
        return {}

    with open(GOLD_GEN_PATH) as f:
        gold = json.load(f)

    unverified = [g for g in gold if not g.get("manual_verified", False)]
    if unverified:
        log.warning(
            "⚠ %d / %d generation gold entries have manual_verified=false.",
            len(unverified), len(gold),
        )

    p1_faith, p2_faith = [], []
    p1_cite_acc, p2_cite_acc = [], []
    per_query = []

    for item in gold:
        query = item["query"]

        # Phase 1: RRF → LLM (no reranker, no NLI guard)
        rrf_res      = retriever.rrf_search(query, top_k=DENSE_TOP_K)
        p1_chunks    = rrf_res[:RERANK_TOP_K]
        p1_answer    = generator.answer_query(query, p1_chunks)
        p1_faith_res = guard.check(p1_answer, p1_chunks)
        p1_cite      = verify_citations(p1_answer, p1_chunks)

        # Phase 2: RRF → reranker → LLM → NLI guard
        p2_chunks    = reranker.rerank(query, rrf_res, top_k=RERANK_TOP_K)
        p2_answer    = generator.answer_query(query, p2_chunks)
        p2_faith_res = guard.check(p2_answer, p2_chunks)
        p2_cite      = verify_citations(p2_answer, p2_chunks)

        p1_faith.append(p1_faith_res["faithfulness"])
        p2_faith.append(p2_faith_res["faithfulness"])

        if p1_cite["citations_present"]:
            p1_cite_acc.append(p1_cite["citation_accuracy"])
        if p2_cite["citations_present"]:
            p2_cite_acc.append(p2_cite["citation_accuracy"])

        per_query.append({
            "query":                  query,
            "p1_faithfulness":        p1_faith_res["faithfulness"],
            "p2_faithfulness":        p2_faith_res["faithfulness"],
            "p1_flagged_sentences":   len(p1_faith_res["flagged_sentences"]),
            "p2_flagged_sentences":   len(p2_faith_res["flagged_sentences"]),
            "p1_citation_accuracy":   p1_cite.get("citation_accuracy"),
            "p2_citation_accuracy":   p2_cite.get("citation_accuracy"),
            "p1_invalid_citations":   p1_cite["invalid_citations"],
            "p2_invalid_citations":   p2_cite["invalid_citations"],
        })

        log.info(
            "[%s...] P1_faith=%.3f P2_faith=%.3f | "
            "P1_cite=%s P2_cite=%s",
            query[:40],
            p1_faith_res["faithfulness"], p2_faith_res["faithfulness"],
            f"{p1_cite['citation_accuracy']:.2f}" if p1_cite["citations_present"] else "n/a",
            f"{p2_cite['citation_accuracy']:.2f}" if p2_cite["citations_present"] else "n/a",
        )

    return {
        "n_queries":               len(gold),
        "unverified":              len(unverified),
        "phase1_avg_faithfulness": round(float(np.mean(p1_faith)), 4),
        "phase2_avg_faithfulness": round(float(np.mean(p2_faith)), 4),
        "delta_faithfulness":      round(float(np.mean(p2_faith)) - float(np.mean(p1_faith)), 4),
        "phase1_avg_citation_acc": round(float(np.mean(p1_cite_acc)), 4) if p1_cite_acc else None,
        "phase2_avg_citation_acc": round(float(np.mean(p2_cite_acc)), 4) if p2_cite_acc else None,
        "per_query":               per_query,
    }


# ── Section 3: Compliance gold evaluation ────────────────────────────────────

def eval_compliance_gold(generator: ComplianceGenerator) -> Dict:
    """
    Evaluate the compliance checker against manually authored gold cases.
    Each case has a policy excerpt and an expected verdict.
    Computes accuracy and per-class precision/recall.

    This is the only correct way to evaluate the compliance checker.
    Running it on real PDFs without ground truth tells you nothing.
    """
    if not GOLD_COMP_PATH.exists():
        log.error("Compliance gold not found: %s — run build_gold.py first.", GOLD_COMP_PATH)
        return {}

    with open(GOLD_COMP_PATH) as f:
        gold = json.load(f)

    from src.kg.deontic_kg import DeonticKG
    deontic_kg = DeonticKG()

    y_true, y_pred = [], []
    per_case = []

    for case in gold:
        rule_subject = case["rule_subject"]
        policy_text  = case["test_policy_excerpt"]
        expected     = case["expected_verdict"]

        rule_triples = deontic_kg.get_relevant_triples(rule_subject)
        if not rule_triples:
            log.warning("No triples for rule_subject: %s", rule_subject)
            continue

        # Build a fake single-chunk context from the policy excerpt
        fake_chunk = [{"id": 0, "filename": "gold_test", "text": policy_text}]

        result  = generator.check_compliance_for_rule(
            rule_subject=rule_subject,
            rule_triples=rule_triples,
            relevant_chunks=fake_chunk,
            policy_text=policy_text,
        )
        predicted = result["verdict"]

        y_true.append(expected)
        y_pred.append(predicted)

        correct = predicted == expected
        per_case.append({
            "rule_subject": rule_subject,
            "expected":     expected,
            "predicted":    predicted,
            "correct":      correct,
            "reason":       result["reason"],
        })
        sym = "✓" if correct else "✗"
        log.info("[%s] %s expected=%s predicted=%s", sym, rule_subject, expected, predicted)

    accuracy = sum(y == p for y, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0

    # Per-class precision and recall
    classes = sorted(set(y_true + y_pred))
    class_metrics = {}
    for cls in classes:
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p == cls)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != cls and p == cls)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[cls] = {
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "support":   sum(1 for y in y_true if y == cls),
        }

    return {
        "n_cases":       len(gold),
        "accuracy":      round(accuracy, 4),
        "class_metrics": class_metrics,
        "per_case":      per_case,
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(ret: Dict, gen: Dict, comp: Dict):
    sep = "=" * 72
    print(f"\n{sep}")
    print(" POLICY-RAG v2 — EVALUATION REPORT")
    print(sep)

    if ret:
        unverif_note = (
            f"  ⚠ {ret['unverified']}/{ret['n_queries']} entries not manually verified"
            if ret.get("unverified", 0) > 0 else ""
        )
        print(f"\n── RETRIEVAL ABLATION  ({ret['usable_queries']} queries){unverif_note}")
        header = f"{'Method':<14}" + "".join(
            f"{'R@1':>6}{'R@3':>6}{'R@5':>6}{'R@10':>7}{'MRR':>7}{'nDCG@10':>9}"
        )
        print(header)
        print("─" * len(header))
        for m, vals in ret["methods"].items():
            row = (
                f"{m:<14}"
                f"{vals.get('recall@1',  0):>6.3f}"
                f"{vals.get('recall@3',  0):>6.3f}"
                f"{vals.get('recall@5',  0):>6.3f}"
                f"{vals.get('recall@10', 0):>7.3f}"
                f"{vals.get('mrr',       0):>7.3f}"
                f"{vals.get('ndcg@10',   0):>9.3f}"
            )
            print(row)

    if gen:
        p1 = gen["phase1_avg_faithfulness"]
        p2 = gen["phase2_avg_faithfulness"]
        d  = gen["delta_faithfulness"]
        print(f"\n── HALLUCINATION + CITATION  ({gen['n_queries']} queries)")
        print(f"  Phase 1 faithfulness    : {p1:.4f}")
        print(f"  Phase 2 faithfulness    : {p2:.4f}")
        print(f"  Δ faithfulness          : {d:+.4f}  ({'improved' if d > 0 else 'degraded'})")
        if gen["phase1_avg_citation_acc"] is not None:
            print(f"  Phase 1 citation acc    : {gen['phase1_avg_citation_acc']:.4f}")
        if gen["phase2_avg_citation_acc"] is not None:
            print(f"  Phase 2 citation acc    : {gen['phase2_avg_citation_acc']:.4f}")

    if comp:
        print(f"\n── COMPLIANCE GOLD  ({comp['n_cases']} cases, manually authored)")
        print(f"  Accuracy: {comp['accuracy']:.4f}")
        print(f"  {'Verdict':<28} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>8}")
        print(f"  {'─'*28} {'─'*10} {'─'*8} {'─'*6} {'─'*8}")
        for cls, m in comp["class_metrics"].items():
            print(f"  {cls:<28} {m['precision']:>10.3f} {m['recall']:>8.3f} "
                  f"{m['f1']:>6.3f} {m['support']:>8}")

    print(f"\n{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-generation",  action="store_true")
    ap.add_argument("--skip-compliance",  action="store_true")
    ap.add_argument("--out",              type=str, default=None)
    args = ap.parse_args()

    log.info("Loading retriever...")
    retriever = UnifiedRetriever()
    log.info("Loading reranker...")
    reranker  = CrossEncoderReranker()

    log.info("=== Section 1: Retrieval ablation ===")
    ret_results = eval_retrieval(retriever, reranker)

    gen_results, comp_results = {}, {}

    # Generator is needed for either generation eval or compliance eval.
    # Load it once here so --skip-generation does not also silence compliance.
    need_generator = not args.skip_generation or not args.skip_compliance
    generator = ComplianceGenerator() if need_generator else None
    if need_generator:
        log.info("Generator loaded")

    if not args.skip_generation:
        log.info("Loading NLI guard...")
        guard = FaithfulnessGuard()
        log.info("=== Section 2: Generation + hallucination + citation ===")
        gen_results = eval_generation(retriever, reranker, guard, generator)
    else:
        log.info("Skipping generation eval (--skip-generation)")

    if not args.skip_compliance:
        log.info("=== Section 3: Compliance gold evaluation ===")
        comp_results = eval_compliance_gold(generator)
    else:
        log.info("Skipping compliance eval (--skip-compliance)")

    print_report(ret_results, gen_results, comp_results)

    if args.out:
        full = {
            "retrieval":   ret_results,
            "generation":  gen_results,
            "compliance":  comp_results,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(full, f, indent=2)
        log.info("Full report saved → %s", out_path)


if __name__ == "__main__":
    main()
