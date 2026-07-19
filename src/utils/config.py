"""
config.py — Central configuration. All paths and constants live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent.parent.resolve()

# ── Data paths ────────────────────────────────────────────────────────────────
DOCS_DIR      = ROOT / "data" / "docs"
INDEX_DIR     = ROOT / "data" / "index"
KG_DIR        = ROOT / "data" / "kg"
EVAL_DIR      = ROOT / "data" / "eval"

CHUNKS_PATH   = INDEX_DIR / "chunks.json"
FAISS_PATH    = INDEX_DIR / "faiss.index"
KG_TRIPLES    = KG_DIR   / "dpdp_triples.json"
GOLD_PATH     = EVAL_DIR / "gold_retrieval.json"
GOLD_GEN_PATH = EVAL_DIR / "gold_generation.json"

# ── Models ────────────────────────────────────────────────────────────────────
# BGE-small outperforms all-MiniLM-L6-v2 on legal/domain-specific text
# (MTEB benchmark legal subsets, 2024). Same size, significantly better.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# BGE reranker is trained on diverse text including legal/technical corpora,
# unlike ms-marco which is purely web search. Better fit for DPDP compliance.
RERANKER_MODEL = "BAAI/bge-reranker-base"

# NLI faithfulness model (label order: contradiction=0, entailment=1, neutral=2)
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

# ── Chunking ──────────────────────────────────────────────────────────────────
MAX_WORDS     = 180
OVERLAP_WORDS = 25

# ── Retrieval ─────────────────────────────────────────────────────────────────
DENSE_TOP_K  = 20
SPARSE_TOP_K = 20
RERANK_TOP_K = 5
GRAPH_HOPS   = 2

# RRF constant — standard 60; tunable here without touching retriever.py
RRF_K = 60

# BM25 parameters — tuned for long legal clauses.
# Default k1=1.5, b=0.75 is calibrated for newswire (short docs, uniform length).
# Legal clauses are long and length varies hugely. Lower b reduces length bias.
# k1=1.2 (moderate term saturation), b=0.4 (weak length normalisation).
BM25_K1 = 1.2
BM25_B  = 0.4

# ── Compliance ────────────────────────────────────────────────────────────────
COMPLIANCE_TOP_K = 5

# Severity weights for compliance scoring (used in run_compliance.py report)
# Keys must match deontic_kg.json "object" values or rule subjects.
SEVERITY_WEIGHTS = {
    "Critical": ["Consent", "Data Breach", "Security Safeguards",
                 "Children Data", "Sensitive Personal Data"],
    "High":     ["Notice", "Purpose Limitation", "Storage Limitation",
                 "Data Minimization", "Cross Border Transfer"],
    "Medium":   ["Data Principal", "Grievance Redressal",
                 "Third Party Sharing", "Automated Decision Making"],
    "Low":      ["Consent Manager", "Data Protection Board",
                 "Significant Data Fiduciary", "Lawful Processing",
                 "Exemption", "Appeal", "Complaint"],
}

# ── Faithfulness guard ────────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD = 0.30

# ── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "groq":   "llama-3.1-8b-instant",
}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULTS.get(LLM_PROVIDER, "gpt-4o-mini")


def get_llm_client():
    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        return _OpenAIClient(OpenAI(api_key=os.getenv("OPENAI_API_KEY")), LLM_MODEL)
    elif LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        return _GeminiClient(genai, LLM_MODEL)
    elif LLM_PROVIDER == "groq":
        from openai import OpenAI
        return _OpenAIClient(
            OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"),
            LLM_MODEL,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Use openai | gemini | groq")


class _OpenAIClient:
    def __init__(self, client, model):
        self.client = client
        self.model  = model

    def chat(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=512,
        )
        return resp.choices[0].message.content.strip()


class _GeminiClient:
    def __init__(self, genai, model):
        self.model = genai.GenerativeModel(model)

    def chat(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        full = (system + "\n\n" + prompt).strip() if system else prompt
        resp = self.model.generate_content(
            full, generation_config={"temperature": temperature, "max_output_tokens": 512},
        )
        return resp.text.strip()
