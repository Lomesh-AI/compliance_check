"""
retriever.py — BM25 + Dense FAISS + KG expansion + RRF fusion.

Fixes in this version:
  - BM25 uses tuned k1=BM25_K1, b=BM25_B from config (legal-doc optimised)
  - BM25 tokenisation strips punctuation (consent. == consent)
  - Graph entity detection uses word-boundary regex (data != database)
  - RRF_K imported from config.py
  - BGE embedding model now used (configured in config.py)
  - BGE requires prepending "Represent this sentence for searching relevant passages:"
    to queries (not passages) per BGE docs. Applied in dense_search only.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import faiss
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import (
    CHUNKS_PATH, FAISS_PATH, KG_TRIPLES, EMBED_MODEL,
    DENSE_TOP_K, SPARSE_TOP_K, RERANK_TOP_K, GRAPH_HOPS,
    RRF_K, BM25_K1, BM25_B,
)

log = logging.getLogger("retriever")

# BGE models require this prefix on QUERIES (not on passages) for retrieval tasks.
# Source: BAAI/bge-small-en-v1.5 model card on HuggingFace.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _tokenize(text: str) -> List[str]:
    """Punctuation-stripped tokenisation for BM25 (consent. == consent)."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class UnifiedRetriever:

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        faiss_path:  Path = FAISS_PATH,
        kg_path:     Path = KG_TRIPLES,
    ):
        self.chunks_path = Path(chunks_path)
        self.faiss_path  = Path(faiss_path)
        self.kg_path     = Path(kg_path)

        self.chunks:     List[Dict]               = []
        self.chunk_map:  Dict[int, Dict]          = {}
        self.index:      Optional[faiss.Index]    = None
        self.bm25:       Optional[BM25Okapi]      = None
        self.model:      Optional[SentenceTransformer] = None
        self.kg:         Optional[nx.DiGraph]     = None
        self.entity_set: set                      = set()

        self._load_chunks()
        self._load_faiss()
        self._build_bm25()
        self._load_embedder()
        self._load_kg()

    def _load_chunks(self):
        if not self.chunks_path.exists():
            raise FileNotFoundError(
                f"chunks.json not found at {self.chunks_path}. "
                "Run: python src/ingestion/ingest.py"
            )
        with open(self.chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["id"]: c for c in self.chunks}
        docs = {c["filename"] for c in self.chunks}
        log.info("Loaded %d chunks from %d doc(s): %s", len(self.chunks), len(docs), docs)

    def _load_faiss(self):
        if not self.faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.faiss_path}. "
                "Run: python src/ingestion/ingest.py"
            )
        self.index = faiss.read_index(str(self.faiss_path))
        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"FAISS DESYNC: {self.index.ntotal} vectors vs {len(self.chunks)} chunks. "
                "Re-run: python src/ingestion/ingest.py"
            )
        log.info("FAISS index: %d vectors ✓", self.index.ntotal)

    def _build_bm25(self):
        corpus = [_tokenize(c["text"]) for c in self.chunks]
        # Tuned parameters for legal text (see config.py for rationale)
        self.bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
        log.info("BM25 index: %d docs | k1=%.1f b=%.2f", len(corpus), BM25_K1, BM25_B)

    def _load_embedder(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        log.info("Embedder: %s", EMBED_MODEL)

    def _load_kg(self):
        if not self.kg_path.exists():
            log.warning("KG not found at %s — graph expansion disabled", self.kg_path)
            return
        with open(self.kg_path, encoding="utf-8") as f:
            triples = json.load(f)
        self.kg = nx.DiGraph()
        for t in triples:
            s = t.get("subject", "").strip()
            o = t.get("object", "").strip()
            p = t.get("predicate", "relatedTo").strip()
            if s and o:
                self.kg.add_edge(s, o, predicate=p)
                self.entity_set.add(s.lower())
                self.entity_set.add(o.lower())
        log.info("KG: %d nodes, %d edges", self.kg.number_of_nodes(), self.kg.number_of_edges())

    # ── Individual retrievers ─────────────────────────────────────────────────

    def dense_search(self, query: str, k: int = DENSE_TOP_K) -> List[Dict]:
        """BGE requires the query prefix for retrieval tasks (not passages)."""
        prefixed = BGE_QUERY_PREFIX + query
        qvec = self.model.encode([prefixed], convert_to_numpy=True, normalize_embeddings=True)
        scores, ids = self.index.search(qvec, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunk_map[idx].copy()
            c["score"]  = float(score)
            c["method"] = "dense"
            results.append(c)
        return results

    def sparse_search(self, query: str, k: int = SPARSE_TOP_K) -> List[Dict]:
        tokenized = _tokenize(query)
        scores    = self.bm25.get_scores(tokenized)
        top_ids   = np.argsort(scores)[::-1][:k]
        results   = []
        for idx in top_ids:
            if scores[idx] <= 0:
                continue
            c = self.chunks[idx].copy()
            c["score"]  = float(scores[idx])
            c["method"] = "sparse"
            results.append(c)
        return results

    def graph_search(self, query: str, k: int = RERANK_TOP_K * 2) -> List[Dict]:
        """Word-boundary entity detection to avoid 'data' matching 'database'."""
        if self.kg is None:
            return []
        query_lower = query.lower()
        matched = {e for e in self.entity_set
                   if re.search(r"\b" + re.escape(e) + r"\b", query_lower)}
        if not matched:
            return []
        expanded = set(matched)
        frontier = set(matched)
        for _ in range(GRAPH_HOPS):
            nxt = set()
            for node in self.kg.nodes():
                if node.lower() in expanded:
                    for nbr in self.kg.successors(node):
                        nxt.add(nbr.lower())
                    for nbr in self.kg.predecessors(node):
                        nxt.add(nbr.lower())
            new = nxt - expanded
            if not new:
                break
            expanded |= new
            frontier = new

        scored = []
        for c in self.chunks:
            text_lower = c["text"].lower()
            hits = sum(1 for e in expanded
                       if re.search(r"\b" + re.escape(e) + r"\b", text_lower))
            if hits > 0:
                cc = c.copy()
                cc["score"]  = float(hits)
                cc["method"] = "graph"
                scored.append(cc)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def rrf_search(self, query: str, top_k: int = DENSE_TOP_K) -> List[Dict]:
        dense_res  = self.dense_search(query, k=top_k)
        sparse_res = self.sparse_search(query, k=top_k)
        graph_res  = self.graph_search(query, k=top_k // 2)

        rrf_scores: Dict[int, float] = {}
        chunk_store: Dict[int, Dict] = {}

        def _add(ranked_list):
            for rank, item in enumerate(ranked_list):
                cid = item["id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
                if cid not in chunk_store:
                    chunk_store[cid] = item

        _add(dense_res)
        _add(sparse_res)
        _add(graph_res)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, rrf_score in ranked[:top_k]:
            c = chunk_store[cid].copy()
            c["rrf_score"] = rrf_score
            c["method"]    = "rrf"
            results.append(c)
        return results

    def verify(self) -> bool:
        ok = self.index is not None and self.index.ntotal == len(self.chunks) and self.bm25 is not None
        log.info("Retriever verify: %s", "PASS ✓" if ok else "FAIL ✗")
        return ok
