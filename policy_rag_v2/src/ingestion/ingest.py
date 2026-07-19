"""
ingest.py — Semantic-boundary chunker + FAISS builder.

Key guarantee: chunk id == FAISS vector position, always.
The old codebase had a desync bug because they were built separately.
Here both are built in ONE pass from the same chunk list, then verified.

Run:
    python src/ingestion/ingest.py
"""

import re
import json
import logging
from pathlib import Path

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import (
    DOCS_DIR, CHUNKS_PATH, FAISS_PATH, INDEX_DIR,
    EMBED_MODEL, MAX_WORDS, OVERLAP_WORDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest")


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    """Extract text from a PDF file page by page."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t.strip())
    return "\n\n".join(pages)


# ── Sentence-aware chunker ───────────────────────────────────────────────────

def _split_sentences(text: str):
    """Split on sentence boundaries using punctuation heuristic."""
    # Split on .  !  ?  ;  followed by whitespace+capital or end
    parts = re.split(r'(?<=[.!?;])\s+(?=[A-Z\(\"\']|\d)', text)
    # Further clean
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_words: int = MAX_WORDS, overlap_words: int = OVERLAP_WORDS):
    """
    Sentence-boundary-aware chunking.
    1. Split into paragraphs (blank lines).
    2. For each paragraph, accumulate sentences until max_words.
    3. On overflow, start a new chunk with overlap_words carried forward.

    This ensures sentence integrity and prevents cutting mid-sentence —
    which was a problem in the old fixed-size word chunker.
    """
    # Split into rough paragraph blocks
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []

    for para in paragraphs:
        sentences = _split_sentences(para)
        if not sentences:
            continue

        buffer = []          # list of sentences in current chunk
        buffer_words = 0

        for sent in sentences:
            sent_words = len(sent.split())

            # If a single sentence exceeds max_words, hard-split it
            if sent_words > max_words:
                if buffer:
                    chunks.append(" ".join(buffer))
                    buffer, buffer_words = [], 0
                words = sent.split()
                for i in range(0, len(words), max_words - overlap_words):
                    slice_words = words[i: i + max_words]
                    if slice_words:
                        chunks.append(" ".join(slice_words))
                continue

            if buffer_words + sent_words > max_words:
                # Flush current buffer
                chunks.append(" ".join(buffer))
                # Carry last N words as overlap
                carry = " ".join(buffer).split()[-overlap_words:]
                buffer = carry + [sent] if carry else [sent]
                buffer_words = len(" ".join(buffer).split())
            else:
                buffer.append(sent)
                buffer_words += sent_words

        if buffer:
            chunks.append(" ".join(buffer))

    return [c for c in chunks if len(c.split()) >= 5]  # discard very short noise


# ── Main build function ───────────────────────────────────────────────────────

def build_index(docs_dir: Path = DOCS_DIR):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect documents
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")
    log.info("Found %d PDFs: %s", len(pdf_files), [f.name for f in pdf_files])

    # 2. Extract and chunk — one pass, one list
    all_chunks = []
    for doc_id, pdf_path in enumerate(pdf_files):
        log.info("Extracting: %s", pdf_path.name)
        text = extract_pdf(pdf_path)
        if not text.strip():
            log.warning("Empty text from %s — skipping", pdf_path.name)
            continue
        raw_chunks = chunk_text(text)
        log.info("  → %d chunks", len(raw_chunks))
        for chunk_text_str in raw_chunks:
            all_chunks.append({
                "id":         len(all_chunks),   # chunk_id == FAISS position
                "doc_id":     doc_id,
                "filename":   pdf_path.name,
                "text":       chunk_text_str,
                "word_count": len(chunk_text_str.split()),
            })

    if not all_chunks:
        raise RuntimeError("No chunks produced. Check PDFs.")

    log.info("Total chunks: %d", len(all_chunks))

    # 3. Embed
    log.info("Loading embedder: %s", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    log.info("Encoding %d chunks...", len(texts))
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)

    # 4. Normalize + build FAISS (IndexFlatIP == cosine after L2-norm)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 5. Save — BOTH at the same time from the same list
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, str(FAISS_PATH))

    # 6. Verify sync (the critical check the old code was missing)
    assert index.ntotal == len(all_chunks), (
        f"SYNC ERROR: FAISS has {index.ntotal} vectors but chunks.json has {len(all_chunks)} entries. "
        "This should never happen in this codebase."
    )
    log.info("✓ VERIFIED: %d chunks == %d FAISS vectors", len(all_chunks), index.ntotal)
    log.info("Saved chunks → %s", CHUNKS_PATH)
    log.info("Saved index  → %s", FAISS_PATH)

    return all_chunks, index


if __name__ == "__main__":
    build_index()
