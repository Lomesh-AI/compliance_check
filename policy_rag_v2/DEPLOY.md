# PolicyRAG v2 — Local Deployment Guide

## Prerequisites

- Python 3.10 or higher
- pip
- ~4 GB disk (models download on first run)
- No GPU needed — all models run on CPU

---

## Step 0 — Clone / unzip and enter the project

```bash
cd policy_rag_v2
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all ML libraries, Flask, BM25, FAISS-CPU, and sentence-transformers.
First run downloads ~400 MB of model weights (BGE small, BGE reranker, DeBERTa NLI).

---

## Step 2 — Set your LLM API key

```bash
cp .env.example .env
```

Edit `.env` and add ONE provider key:

```bash
# Recommended (free): Groq
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq

# Or OpenAI
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai

# Or Gemini
GEMINI_API_KEY=AI...
LLM_PROVIDER=gemini
```

Groq is recommended for testing — free tier, fast, supports llama3-8b-8192.
Sign up at https://console.groq.com

---

## Step 3 — Build the index

**Must run this before anything else.**

```bash
python src/ingestion/ingest.py
```

Reads all PDFs from `data/docs/`, creates sentence-boundary chunks, builds
FAISS index, and verifies sync. Takes ~30–60 seconds on CPU.

Output:
- `data/index/chunks.json` — all chunks with stable IDs
- `data/index/faiss.index` — FAISS IndexFlatIP

---

## Step 4 — Build gold evaluation sets

```bash
python src/eval/build_gold.py
```

Creates:
- `data/eval/gold_retrieval.json` — 50 DPDP retrieval queries
- `data/eval/gold_generation.json` — 15 QA pairs with gold answers
- `data/eval/gold_compliance.json` — 10 manually authored compliance test cases

**After running**, open `data/eval/gold_retrieval.json` and manually verify
the `relevant_chunk_ids` for each entry. Set `"manual_verified": true` on
entries you have checked. The eval harness warns on unverified entries.

---

## Step 5 — Run the evaluation harness

```bash
# Full evaluation (retrieval ablation + hallucination + compliance gold)
python src/eval/run_eval.py

# Retrieval ablation only (no LLM calls — fast)
python src/eval/run_eval.py --skip-generation

# Save full JSON report
python src/eval/run_eval.py --out data/eval/report.json
```

The ablation table shows:

| Method      | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|-------------|-----|-----|-----|------|-----|---------|
| dense       | ... | ... | ... | ...  | ... | ...     |
| sparse      | ... | ... | ... | ...  | ... | ...     |
| rrf         | ... | ... | ... | ...  | ... | ...     |
| rrf_rerank  | ... | ... | ... | ...  | ... | ...     |

This proves each component adds value (or doesn't).

---

## Step 6 — Compliance check on a business PDF

```bash
# Indian Oil privacy policy
python scripts/run_compliance.py --pdf "data/docs/Indian Oil Priivacy policy.pdf"

# HDFC privacy policy — save results to JSON
python scripts/run_compliance.py --pdf "data/docs/Privacy_Policy_hdfc.pdf" --out results.json

# Without cross-encoder reranker (faster, less accurate)
python scripts/run_compliance.py --pdf "data/docs/Privacy_Policy_hdfc.pdf" --no-rerank

# Without LLM cache (always makes fresh API calls)
python scripts/run_compliance.py --pdf "data/docs/Privacy_Policy_hdfc.pdf" --no-cache
```

Output shows:
- Per-rule verdict (PASS / FAIL / INSUFFICIENT_EVIDENCE) with severity tier
- How many real LLM calls were made vs cache hits vs pre-screen skips
- Document-level overall verdict with severity-weighted scoring

---

## Step 7 — Interactive DPDP Act QA

```bash
# Interactive mode
python scripts/ask.py

# Single query
python scripts/ask.py --query "What are the rights of a Data Principal?"

# Without NLI faithfulness guard (faster)
python scripts/ask.py --query "What must a Data Fiduciary do after a breach?" --no-guard

# Disable query logging
python scripts/ask.py --no-log
```

Queries and answers are logged to `data/eval/query_log.jsonl` by default.

---

## Step 8 — Run the API server

```bash
python src/api/app.py
```

Server starts at http://localhost:5000

API endpoints:
- `GET  /api/health`       — server status
- `GET  /api/kg/summary`   — deontic KG statistics
- `GET  /api/kg/rules`     — all rules (optional `?subject=Consent`)
- `POST /api/ask`          — DPDP Act QA (`{"query": "..."}`)
- `POST /api/check`        — compliance check (multipart PDF upload)

---

## Step 9 — Open the frontend

With the server running at localhost:5000, open the frontend in your browser:

```bash
# Option A: open file directly (works for most browsers)
open frontend/index.html

# Option B: serve it via Python (avoids any CORS issues)
python -m http.server 8080 --directory frontend
# then open http://localhost:8080
```

The frontend has three pages:
- **Compliance Check** — upload a PDF, see per-rule PASS/FAIL with severity
- **Ask DPDP Act** — type a question, get a grounded answer with faithfulness score
- **KG Explorer** — browse all 65 deontic rules, filter by subject or type

---

## Adding more PDF documents

1. Drop the PDF into `data/docs/`
2. Re-run ingestion: `python src/ingestion/ingest.py`
3. Re-run gold build: `python src/eval/build_gold.py`

Business policy PDFs (Indian Oil, HDFC, etc.) will appear in compliance checks.
Regulatory PDFs (DPDP Act, GDPR) will appear in the QA index.
The system automatically separates them by filename.

See `DATA_SOURCES.md` for a list of recommended additional documents.

---

## Troubleshooting

**`FileNotFoundError: chunks.json not found`**
→ Run `python src/ingestion/ingest.py` first.

**`FAISS DESYNC`**
→ Re-run `python src/ingestion/ingest.py`. The index and chunks got out of sync
(this should not happen with the current code but can if you manually edited files).

**`Model loading failed` on server start**
→ Run ingestion first. The server loads the FAISS index at startup.

**`429 Too Many Requests` from Groq**
→ Groq free tier limits to ~30 RPM. The compliance checker makes at most 19 real
LLM calls (one per rule-group). If you hit limits, add `--no-cache` to disable
the cache and retry with `--no-rerank` for faster throughput.

**Models downloading very slowly**
→ BGE-small (~130 MB), BGE-reranker-base (~1.1 GB), DeBERTa-NLI (~180 MB).
Total ~1.4 GB on first run. Models are cached by sentence-transformers in
`~/.cache/huggingface/` after the first download.

**Frontend shows "server offline"**
→ Make sure `python src/api/app.py` is running in a separate terminal.

---

## Full command sequence (copy-paste)

```bash
# One-time setup
pip install -r requirements.txt
cp .env.example .env
# edit .env with your GROQ_API_KEY

# Build pipeline
python src/ingestion/ingest.py
python src/eval/build_gold.py

# Evaluate
python src/eval/run_eval.py --skip-generation   # fast retrieval ablation
python src/eval/run_eval.py                      # full eval (LLM calls needed)

# Use
python scripts/run_compliance.py --pdf "data/docs/Privacy_Policy_hdfc.pdf"
python scripts/ask.py

# API + Frontend
python src/api/app.py &          # start API in background
python -m http.server 8080 --directory frontend
# open http://localhost:8080
```
