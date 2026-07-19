"""
run_compliance.py — Rule-by-rule compliance check using the full retrieval pipeline.

PIPELINE (19 LLM calls max, not one per chunk):
  For each KG rule-group (subject → triples):
    1. detect_gaps() pre-screens the rule against retrieved chunk text (rule-based, free)
    2. If pre-screen gives CONFIDENT result → SKIP LLM CALL
    3. Otherwise → RRF retrieval → rerank → LLM → verdict
  Final: document-level report with severity-weighted scoring.

Fixes in this version:
  - LLM call counter tracks ACTUAL API calls, not cache hits (#1)
  - detect_gaps() now GATES LLM calls: skips when pre-screen is confident (#2)
  - build_retrieval_query() produces natural language, not a bag of words (#3)
  - Severity weighting from config.SEVERITY_WEIGHTS (#11)
  - Cache hits correctly reported as "cache hit", not counted as new calls

Usage:
    python scripts/run_compliance.py --pdf "data/docs/Indian Oil Priivacy policy.pdf"
    python scripts/run_compliance.py --pdf "data/docs/Privacy_Policy_hdfc.pdf" --out results.json
    python scripts/run_compliance.py --pdf "data/docs/..." --no-rerank
    python scripts/run_compliance.py --pdf "data/docs/..." --no-cache
"""

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.ingest import extract_pdf, chunk_text
from src.retrieval.retriever import UnifiedRetriever, _tokenize
from src.reranker.reranker import CrossEncoderReranker
from src.generation.generator import ComplianceGenerator
from src.kg.deontic_kg import DeonticKG
from src.utils.config import COMPLIANCE_TOP_K, DENSE_TOP_K, SEVERITY_WEIGHTS

log = logging.getLogger("run_compliance")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Severity helper ───────────────────────────────────────────────────────────

def _get_severity(rule_subject: str) -> str:
    for level, subjects in SEVERITY_WEIGHTS.items():
        if rule_subject in subjects:
            return level
    return "Low"


# ── Natural-language retrieval query builder ──────────────────────────────────

def build_retrieval_query(subject: str, triples: List[Dict]) -> str:
    """
    Produce a natural-language query from a KG rule-group.

    FIX: old version joined objects with spaces → bag-of-words "Consent Free Specific".
    New version constructs a grammatical sentence using the predicate structure.

    Examples:
      Consent + [mustBe Free, mustBe Specific, mustBe Informed]
        → "Consent must be free, specific, and informed"
      Data Fiduciary + [mustObtain Consent, mustProvide Notice]
        → "Data Fiduciary must obtain consent and provide notice"
    """
    # Group by predicate family (mustBe, mustObtain, hasRightTo, etc.)
    pred_groups: Dict[str, List[str]] = defaultdict(list)
    for t in triples:
        pred = t.get("predicate", "")
        obj  = t.get("object", "")
        if obj:
            pred_groups[pred].append(obj.lower())

    if not pred_groups:
        return subject

    clauses = []
    for pred, objects in pred_groups.items():
        # Convert camelCase predicate to readable words
        readable = _predicate_to_words(pred)
        obj_str  = ", ".join(objects[:-1]) + (f", and {objects[-1]}" if len(objects) > 1 else objects[0])
        clauses.append(f"{readable} {obj_str}")

    return f"{subject} " + "; ".join(clauses)


def _predicate_to_words(pred: str) -> str:
    """mustObtain → 'must obtain', hasRightTo → 'has right to'."""
    # Insert space before uppercase letters
    spaced = re.sub(r'([A-Z])', r' \1', pred).strip().lower()
    # Normalise common deontic predicates
    mapping = {
        "must obtain":     "must obtain",
        "must provide":    "must provide",
        "must be":         "must be",
        "must ensure":     "must ensure",
        "must implement":  "must implement",
        "must notify":     "must notify",
        "must maintain":   "must maintain",
        "must delete":     "must delete",
        "must not":        "must not",
        "must not process":"must not process",
        "must not transfer":"must not transfer",
        "has right to":    "has right to",
        "requires":        "requires",
        "prohibits":       "prohibits",
        "permitted to":    "is permitted to",
        "subject to":      "is subject to",
    }
    for k, v in mapping.items():
        if k in spaced:
            return v
    return spaced


import re  # needed for _predicate_to_words — import here to avoid circular


# ── Per-document retrieval index ──────────────────────────────────────────────

class DocumentIndex:
    """
    In-memory RRF index over a single PDF. Does NOT touch the shared index.
    Reason: shared index contains DPDP Act chunks; we only want business policy evidence.
    """

    def __init__(self, pdf_path: Path):
        import faiss as faiss_lib
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        from src.ingestion.ingest import extract_pdf, chunk_text
        from src.utils.config import EMBED_MODEL, BM25_K1, BM25_B
        from src.retrieval.retriever import BGE_QUERY_PREFIX

        log.info("Building in-memory index for: %s", pdf_path.name)
        text = extract_pdf(pdf_path)
        raw  = chunk_text(text)
        self.chunks = [
            {"id": i, "filename": pdf_path.name, "text": t, "word_count": len(t.split())}
            for i, t in enumerate(raw) if t.strip()
        ]
        log.info("  %d chunks extracted", len(self.chunks))

        # BM25 with tuned legal params
        corpus    = [_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)

        # Dense
        self.embed_model   = SentenceTransformer(EMBED_MODEL)
        self.bge_prefix    = BGE_QUERY_PREFIX
        texts = [c["text"] for c in self.chunks]
        vecs  = self.embed_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        dim  = vecs.shape[1]
        self.faiss_index = faiss_lib.IndexFlatIP(dim)
        self.faiss_index.add(vecs)
        log.info("  Index ready: %d dense vectors, %d BM25 docs", self.faiss_index.ntotal, len(corpus))
        self._np = np

    def search(self, query: str, top_k: int = DENSE_TOP_K) -> List[Dict]:
        from src.utils.config import RRF_K
        np = self._np

        # Dense with BGE query prefix
        qvec   = self.embed_model.encode(
            [self.bge_prefix + query], convert_to_numpy=True, normalize_embeddings=True
        )
        d_scores, d_ids = self.faiss_index.search(qvec, min(top_k, len(self.chunks)))

        # Sparse
        s_scores = self.bm25.get_scores(_tokenize(query))
        s_ranked = np.argsort(s_scores)[::-1][:top_k]

        rrf: Dict[int, float] = {}
        for rank, idx in enumerate(d_ids[0]):
            if idx >= 0:
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(s_ranked):
            if s_scores[idx] > 0:
                rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [
            dict(self.chunks[cid], rrf_score=score)
            for cid, score in ranked[:top_k]
            if cid < len(self.chunks)
        ]

    @property
    def full_text(self) -> str:
        return " ".join(c["text"] for c in self.chunks)


# ── LLM response cache ────────────────────────────────────────────────────────

class _LLMCache:
    def __init__(self, cache_path: Path):
        self.path = cache_path
        self.data: Dict = {}
        if cache_path.exists():
            with open(cache_path) as f:
                self.data = json.load(f)
            log.info("LLM cache: %d entries loaded", len(self.data))

    def _key(self, rule_subject: str, doc_name: str) -> str:
        # Use doc_name + rule_subject instead of context text
        return f"{doc_name}:{rule_subject}"

    def get(self, rule_subject: str, doc_name: str) -> Optional[Dict]:
        return self.data.get(self._key(rule_subject, doc_name))

    def set(self, rule_subject: str, doc_name: str, result: Dict):
        self.data[self._key(rule_subject, doc_name)] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


# ── Main compliance runner ────────────────────────────────────────────────────

def run(
    pdf_path:   Path,
    out_path:   Optional[Path] = None,
    use_rerank: bool = True,
    use_cache:  bool = True,
) -> Dict:
    deontic_kg = DeonticKG()
    by_subject = deontic_kg.by_subject
    log.info("Deontic KG: %d rule-groups | %d triples",
             len(by_subject), len(deontic_kg.static_triples))

    doc_index = DocumentIndex(pdf_path)
    if not doc_index.chunks:
        log.error("No text extracted from %s", pdf_path.name)
        return {}

    reranker  = CrossEncoderReranker() if use_rerank else None
    generator = ComplianceGenerator()

    cache_path = pdf_path.parent.parent / "index" / f"llm_cache_{pdf_path.stem}.json"
    cache      = _LLMCache(cache_path) if use_cache else None

    # Counters — track REAL LLM calls vs skips separately
    real_llm_calls     = 0   # actual API calls made
    cache_hits         = 0   # answered from cache
    prescreen_skips    = 0   # skipped because detect_gaps was confident
    verdict_counts     = defaultdict(int)
    rule_results       = []

    print(f"\nChecking: {pdf_path.name}")
    print(f"Rules: {len(by_subject)}  |  Chunks: {len(doc_index.chunks)}  "
          f"|  Reranker: {'on' if use_rerank else 'off'}\n")

    for rule_subject, rule_triples in sorted(by_subject.items()):
        severity = _get_severity(rule_subject)
        query    = build_retrieval_query(rule_subject, rule_triples)

        if cache:
            cached = cache.get(rule_subject, pdf_path.name)
            if cached:
                result = cached
                cache_hits += 1
                log.info("[%s] Cache hit (skipping retrieval)", rule_subject)
                
                verdict = result["verdict"]
                verdict_counts[verdict] += 1
                sym = {"pass": "✓", "fail": "✗", "insufficient_evidence": "?", "unknown": "!"}.get(verdict, "?")
                print(f"  [{sym}] {rule_subject:<30} [{severity}] → {verdict.upper()} [CACHE]")
                
                rule_results.append({
                    "rule_subject": rule_subject, "severity": severity, "query_used": query,
                    "chunks_retrieved": 0, "top_chunk_ids": [],
                    "verdict": verdict, "reason": result["reason"], "prescreen_skip": False
                })
                continue  # SKIP the rest of the loop for this rule!

        log.info("[%s] [%s] Query: %s", rule_subject, severity, query)



        # Step 1: retrieve relevant chunks
        candidates = doc_index.search(query, top_k=DENSE_TOP_K)
        if reranker and candidates:
            top_chunks = reranker.rerank(query, candidates, top_k=COMPLIANCE_TOP_K)
        else:
            top_chunks = candidates[:COMPLIANCE_TOP_K]

        # Step 2: rule-based pre-screening with synonym normalisation
        chunk_text_combined = " ".join(c["text"] for c in top_chunks)
        gap = deontic_kg.detect_gaps(chunk_text_combined)

        # Step 3: decide whether to skip LLM call
        # Skip if ALL obligations are covered (confident PASS) or violations found (confident FAIL)
        result = None
        skip_reason = ""

        if not top_chunks:
            result = {
                "verdict": "insufficient_evidence",
                "reason":  "No relevant chunks retrieved for this rule.",
                "raw": "", "gap_prescreen": gap,
            }
            skip_reason = "no chunks"

        elif gap["summary"]["violations"] > 0 and gap["summary"]["covered"] == 0:
            # Rule-based: clear violation found, no coverage → confident FAIL
            # Still escalate to LLM to get a human-readable reason
            pass  # let LLM handle this case; we want the reason

        elif (gap["summary"]["covered"] >= len([t for t in rule_triples if t["deontic"] == "obligation"])
              and gap["summary"]["violations"] == 0
              and len([t for t in rule_triples if t["deontic"] == "obligation"]) > 0):
            # All obligations covered, no violations → confident PASS, skip LLM
            result = {
                "verdict": "pass",
                "reason":  f"Rule-based pre-screen: all {gap['summary']['covered']} obligation(s) found in policy text.",
                "raw": "", "gap_prescreen": gap,
            }
            skip_reason = "prescreen_pass"
            prescreen_skips += 1

        # Step 4: LLM call (if not skipped by pre-screening)
        if result is None:
            context_str = " ".join(c["text"] for c in top_chunks)
            cached      = cache.get(rule_subject, context_str) if cache else None

            if cached:
                result     = cached
                cache_hits += 1
                log.info("[%s] Cache hit", rule_subject)
            else:
                result = generator.check_compliance_for_rule(
                    rule_subject=rule_subject,
                    rule_triples=rule_triples,
                    relevant_chunks=top_chunks,
                    policy_text=chunk_text_combined,
                )
                real_llm_calls += 1
                if cache:
                    cache.set(rule_subject, context_str, result)

        verdict = result["verdict"]
        verdict_counts[verdict] += 1

        sym = {"pass": "✓", "fail": "✗", "insufficient_evidence": "?", "unknown": "!"}.get(verdict, "?")
        cache_tag  = " [CACHE]"     if (result is not None and skip_reason == "") and cached else ""
        screen_tag = " [PRESCREEN]" if skip_reason == "prescreen_pass" else ""
        print(f"  [{sym}] {rule_subject:<30} [{severity}] → {verdict.upper()}{cache_tag}{screen_tag}")
        if verdict == "fail":
            print(f"       {result['reason'][:120]}")

        rule_results.append({
            "rule_subject":    rule_subject,
            "severity":        severity,
            "query_used":      query,
            "chunks_retrieved": len(top_chunks),
            "top_chunk_ids":   [c["id"] for c in top_chunks],
            "verdict":         verdict,
            "reason":          result["reason"],
            "prescreen_skip":  skip_reason == "prescreen_pass",
            "gap_summary":     gap.get("summary", {}),
        })

    # ── Document-level verdict with severity weighting ─────────────────────────
    total  = sum(verdict_counts.values())
    fails  = verdict_counts.get("fail", 0)
    passes = verdict_counts.get("pass", 0)
    insuff = verdict_counts.get("insufficient_evidence", 0)

    # Critical failures always make the document FAIL regardless of other rules
    critical_fails = [r for r in rule_results
                      if r["verdict"] == "fail" and r["severity"] == "Critical"]
    high_fails     = [r for r in rule_results
                      if r["verdict"] == "fail" and r["severity"] == "High"]

    if critical_fails:
        doc_verdict = "FAIL (Critical)"
    elif high_fails:
        doc_verdict = "FAIL (High)"
    elif fails > 0:
        doc_verdict = "FAIL"
    elif insuff == 0:
        doc_verdict = "PASS"
    else:
        doc_verdict = "PARTIAL"

    print(f"\n{'='*65}")
    print(f"Document : {pdf_path.name}")
    print(f"Verdict  : {doc_verdict}")
    print(f"{'─'*65}")
    print(f"  PASS                 : {passes}")
    print(f"  FAIL                 : {fails}  (critical={len(critical_fails)} high={len(high_fails)})")
    print(f"  INSUFFICIENT_EVIDENCE: {insuff}")
    print(f"{'─'*65}")
    print(f"  Real LLM calls       : {real_llm_calls}  ← actual API calls")
    print(f"  Cache hits           : {cache_hits}")
    print(f"  Pre-screen skips     : {prescreen_skips}")
    print(f"  Rules total          : {total}")
    print(f"  Doc chunks           : {len(doc_index.chunks)}")
    print(f"{'='*65}\n")

    output = {
        "document":        pdf_path.name,
        "overall_verdict": doc_verdict,
        "verdict_counts":  dict(verdict_counts),
        "llm_calls": {
            "real_api_calls":  real_llm_calls,   # what you actually pay for
            "cache_hits":      cache_hits,
            "prescreen_skips": prescreen_skips,
            "total_rules":     total,
        },
        "doc_chunks":      len(doc_index.chunks),
        "rule_results":    rule_results,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info("Results saved → %s", out_path)

    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DPDP Act compliance check")
    ap.add_argument("--pdf",       required=True)
    ap.add_argument("--out",       default=None)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-cache",  action="store_true")
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"Error: file not found: {pdf}")
        sys.exit(1)

    run(
        pdf_path=pdf,
        out_path=Path(args.out) if args.out else None,
        use_rerank=not args.no_rerank,
        use_cache=not args.no_cache,
    )
