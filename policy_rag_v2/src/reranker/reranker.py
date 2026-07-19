"""
reranker.py — CrossEncoderReranker (BGE) + FaithfulnessGuard (NLI DeBERTa).

Changes from previous version:
  - Model switched to BAAI/bge-reranker-base (trained on diverse text
    including legal/technical corpora). ms-marco was trained on web search only.
  - BGE reranker does NOT require a query prefix (unlike BGE embedder).
    Source: BAAI/bge-reranker-base model card.
  - NLI label order verified: nli-deberta-v3-small outputs
    [contradiction=0, entailment=1, neutral=2]
"""

import logging
import re
from typing import List, Dict

import numpy as np
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import RERANKER_MODEL, NLI_MODEL, RERANK_TOP_K, FAITHFULNESS_THRESHOLD

log = logging.getLogger("reranker")


class CrossEncoderReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-base.

    Scores (query, passage) pairs jointly — full attention between them,
    unlike bi-encoder FAISS which embeds them independently.
    BGE reranker outperforms ms-marco on legal/technical domains.
    CPU inference: ~0.4s per pair.
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        log.info("Loading reranker: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)
        log.info("Reranker ready")

    def rerank(self, query: str, chunks: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
        if not chunks:
            return []
        pairs  = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            [dict(c, rerank_score=float(s)) for c, s in zip(chunks, scores)],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]


class FaithfulnessGuard:
    """
    NLI-based post-generation faithfulness check.

    Label order for cross-encoder/nli-deberta-v3-small:
      output[:, 0] = contradiction
      output[:, 1] = entailment    ← used here
      output[:, 2] = neutral
    Source: https://huggingface.co/cross-encoder/nli-deberta-v3-small
    """

    def __init__(self, model_name: str = NLI_MODEL):
        log.info("Loading NLI faithfulness model: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)
        log.info("Faithfulness guard ready")

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if len(p.strip().split()) >= 4]

    def check(
        self,
        answer: str,
        context_chunks: List[Dict],
        threshold: float = FAITHFULNESS_THRESHOLD,
    ) -> Dict:
        sentences = self._split_sentences(answer)
        if not sentences:
            return {
                "faithfulness": 1.0, "supported_count": 0,
                "total_count": 0, "filtered_answer": answer,
                "flagged_sentences": [],
            }

        context_text = " ".join(c["text"] for c in context_chunks)
        context_text = " ".join(context_text.split()[:1000])

        pairs      = [(context_text, s) for s in sentences]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        if raw_scores.ndim == 2 and raw_scores.shape[1] == 3:
            entail_scores = raw_scores[:, 1]
        else:
            entail_scores = raw_scores.flatten()

        supported, flagged = [], []
        for sent, score in zip(sentences, entail_scores):
            if float(score) >= threshold:
                supported.append(sent)
            else:
                flagged.append({
                    "sentence":          sent,
                    "entailment_score":  round(float(score), 4),
                })

        faithfulness    = len(supported) / len(sentences)
        filtered_answer = " ".join(supported) if supported else "[No grounded sentences]"

        return {
            "faithfulness":      round(faithfulness, 4),
            "supported_count":   len(supported),
            "total_count":       len(sentences),
            "filtered_answer":   filtered_answer,
            "flagged_sentences": flagged,
        }
