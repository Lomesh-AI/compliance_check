# PolicyRAG v2 — DPDP Act Compliance Checker

A multi-stage ML pipeline for checking business privacy policies against India's
Digital Personal Data Protection Act 2023 (DPDP Act).

---

## Project structure

```
policy_rag_v2/
├── data/
│   ├── docs/          ← source PDFs (DPDP Act + business policies)
│   ├── index/         ← built by ingest.py (chunks.json, faiss.index)
│   ├── kg/            ← DPDP Act knowledge graph triples
│   └── eval/          ← built by build_gold.py (gold_retrieval.json, gold_generation.json)
├── src/
│   ├── ingestion/     ingest.py         — PDF → chunks → FAISS
│   ├── retrieval/     retriever.py      — dense + BM25 + graph + RRF
│   ├── reranker/      reranker.py       — cross-encoder reranker + NLI guard
│   ├── generation/    generator.py      — citation-anchored LLM generation
│   ├── eval/          build_gold.py     — build gold sets
│   │                  run_eval.py       — full evaluation harness
│   └── utils/         config.py         — all paths and constants
├── scripts/
│   ├── ask.py                           — interactive DPDP QA
│   └── run_compliance.py               — compliance check on a PDF
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
# 1. Clone / unzip the project
cd policy_rag_v2

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Set your LLM API key
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (free) or OPENAI_API_KEY
```

---

## Run order (must follow this sequence)

### Step 1 — Build the index

Reads all PDFs from `data/docs/`, creates sentence-boundary-aware chunks,
builds a FAISS index, and verifies chunk count == FAISS vector count.

```bash
python src/ingestion/ingest.py
```

Output:
- `data/index/chunks.json`  — all chunks with stable IDs
- `data/index/faiss.index`  — FAISS IndexFlatIP (cosine similarity)

---

### Step 2 — Build gold evaluation sets

Creates clean ground truth for retrieval and generation evaluation.
Requires Step 1 to be done first.

```bash
python src/eval/build_gold.py
```

Output:
- `data/eval/gold_retrieval.json`   — 10 DPDP queries with relevant chunk IDs
- `data/eval/gold_generation.json`  — 8 QA pairs with gold answers from the Act

---

### Step 3 — Run the full evaluation

```bash
# Full evaluation (retrieval + generation + compliance)
python src/eval/run_eval.py

# Retrieval metrics only (no LLM calls — fast)
python src/eval/run_eval.py --skip-generation

# Save report to JSON
python src/eval/run_eval.py --out data/eval/report.json
```

The report shows:
- **Retrieval**: Recall@1/3/5/10, MRR, nDCG@10 for dense / sparse / RRF / RRF+rerank
- **Hallucination**: Phase 1 faithfulness vs Phase 2 faithfulness (Δ)
- **Compliance**: PASS/FAIL distribution across business policy chunks

---

### Step 4 — Ask questions interactively

```bash
# Interactive mode
python scripts/ask.py

# Single query
python scripts/ask.py --query "What are the rights of a Data Principal?"

# Without NLI guard (faster)
python scripts/ask.py --query "..." --no-guard
```

---

### Step 5 — Check a business policy PDF

```bash
python scripts/run_compliance.py --pdf "data/docs/Indian Oil Priivacy policy.pdf"
python scripts/run_compliance.py --pdf "data/docs/Privacy_Policy_hdfc.pdf" --out results.json
```

---

## Pipeline (Phase 2)

```
User query
    │
    ├── Dense retrieval    (FAISS, top-20)
    ├── Sparse retrieval   (BM25, top-20)
    └── Graph expansion    (KG entity walk, DPDP triples)
            │
         RRF Fusion  (Reciprocal Rank Fusion, k=60)
            │
    Cross-encoder Reranker  (ms-marco-MiniLM-L-6-v2, top-5)
            │
    LLM  (citation-anchored prompt: cite [chunk_N] for every claim)
            │
    NLI Faithfulness Guard  (nli-deberta-v3-small, per-sentence entailment)
            │
    Grounded Answer + Faithfulness Score
```

---

## Key differences from the old codebase

| Old codebase | PolicyRAG v2 |
|---|---|
| FAISS and chunks.json built separately → desync bug | Built in one pass, asserted equal |
| OPP-115 US privacy policy chunks mixed into DPDP index | DPDP Act and business PDFs only |
| gold_retrieval_opp115_proxy.json: category-similarity task masquerading as retrieval | Clean 10-query DPDP retrieval gold built with HyDE |
| Linear score interpolation for hybrid | RRF (parameter-free, consistently better) |
| No reranker | Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) |
| No hallucination measurement | NLI faithfulness score before and after |
| Multiple rag_runner*.py, retriever*.py, build_chunk_index*.py files | One canonical file per role |
| Hardcoded Windows paths in eval_mcc.py | All paths from config.py |
| eval/ folder had empty JSON files | Built by build_gold.py |

---

## No GPU needed

All models run on CPU:
- `all-MiniLM-L6-v2` — embedder, ~90MB
- `ms-marco-MiniLM-L-6-v2` — reranker, ~90MB, ~0.3s/pair
- `nli-deberta-v3-small` — NLI guard, ~180MB, ~0.5s/sentence
