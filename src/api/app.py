"""
app.py — Flask REST API for PolicyRAG v2.

Endpoints:
  POST /api/check         — compliance check a single uploaded PDF
  POST /api/ask           — DPDP Act QA
  GET  /api/health        — health check / readiness
  GET  /api/kg/summary    — deontic KG stats
  GET  /api/kg/rules      — all rules grouped by subject

The frontend (frontend/index.html) calls these endpoints directly.

IMPORTANT: Run ingest.py before starting the server.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.retriever import UnifiedRetriever
from src.reranker.reranker import CrossEncoderReranker, FaithfulnessGuard
from src.generation.generator import ComplianceGenerator
from src.kg.deontic_kg import DeonticKG
from src.utils.config import DENSE_TOP_K, RERANK_TOP_K

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")

app = Flask(__name__, static_folder=None)
CORS(app)   # allow frontend on different port during dev

# Add this block to force CORS headers on every response
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# ── Load models once at startup ───────────────────────────────────────────────
log.info("Loading models...")

try:
    _retriever  = UnifiedRetriever()
    _reranker   = CrossEncoderReranker()
    _guard      = FaithfulnessGuard()
    _generator  = ComplianceGenerator()
    _deontic_kg = DeonticKG()
    log.info("All models loaded ✓")
    _ready = True
except Exception as e:
    log.error("Model loading failed: %s", e)
    log.error("Run python src/ingestion/ingest.py first, then restart the server.")
    _ready = False


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    if not _ready:
        return jsonify({"status": "not_ready", "error": "Run ingest.py first"}), 503
    return jsonify({
        "status": "ok",
        "kg_triples": len(_deontic_kg.static_triples),
        "chunks": _retriever.index.ntotal if _retriever.index else 0,
    })


# ── KG info ───────────────────────────────────────────────────────────────────

@app.route("/api/kg/summary")
def kg_summary():
    if not _ready:
        return jsonify({"error": "Server not ready"}), 503
    return jsonify({
        "total_triples": len(_deontic_kg.static_triples),
        "obligations":   len(_deontic_kg.by_deontic.get("obligation", [])),
        "prohibitions":  len(_deontic_kg.by_deontic.get("prohibition", [])),
        "permissions":   len(_deontic_kg.by_deontic.get("permission", [])),
        "entitlements":  len(_deontic_kg.by_deontic.get("entitlement", [])),
        "subjects":      sorted(_deontic_kg.by_subject.keys()),
    })


@app.route("/api/kg/rules")
def kg_rules():
    if not _ready:
        return jsonify({"error": "Server not ready"}), 503
    subject = request.args.get("subject")
    if subject:
        triples = _deontic_kg.get_relevant_triples(subject)
        return jsonify({"subject": subject, "triples": triples})
    # Return all, grouped
    grouped = {}
    for subj, triples in _deontic_kg.by_subject.items():
        grouped[subj] = triples
    return jsonify(grouped)


# ── DPDP QA ───────────────────────────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask():
    if not _ready:
        return jsonify({"error": "Server not ready"}), 503

    data  = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query field is required"}), 400

    try:
        rrf_res    = _retriever.rrf_search(query, top_k=DENSE_TOP_K)
        top_chunks = _reranker.rerank(query, rrf_res, top_k=RERANK_TOP_K)
        raw_answer = _generator.answer_query(query, top_chunks)
        faith      = _guard.check(raw_answer, top_chunks)

        sources = [
            {
                "chunk_id": c.get("id"),
                "filename": c.get("filename"),
                "score":    round(c.get("rerank_score", 0), 4),
                "preview":  c.get("text", "")[:200],
            }
            for c in top_chunks
        ]

        return jsonify({
            "query":              query,
            "answer":             faith["filtered_answer"],
            "raw_answer":         raw_answer,
            "faithfulness":       faith["faithfulness"],
            "supported_count":    faith["supported_count"],
            "total_sentences":    faith["total_count"],
            "flagged_sentences":  faith["flagged_sentences"],
            "sources":            sources,
        })

    except Exception as e:
        log.exception("Error in /api/ask")
        return jsonify({"error": str(e)}), 500


# ── Compliance check ──────────────────────────────────────────────────────────

# @app.route("/api/check", methods=["POST"])
# def check_compliance():
#     if not _ready:
#         return jsonify({"error": "Server not ready"}), 503

#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded. Send a PDF as multipart/form-data field 'file'"}), 400

#     pdf_file = request.files["file"]
#     if not pdf_file.filename.lower().endswith(".pdf"):
#         return jsonify({"error": "Only PDF files are supported"}), 400

#     try:
#         # Save upload to a temp file
#         with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#             pdf_file.save(tmp.name)
#             tmp_path = Path(tmp.name)

#         # Import here to avoid circular import
#         from scripts.run_compliance import run as run_compliance
#         result = run_compliance(
#             pdf_path=tmp_path,
#             out_path=None,
#             use_rerank=True,
#             use_cache=False,   # no cache for fresh uploads
#         )
#         tmp_path.unlink(missing_ok=True)

#         if not result:
#             return jsonify({"error": "No text could be extracted from the PDF"}), 422

#         # Add deontic breakdown to response
#         result["deontic_kg_summary"] = {
#             "total_triples": len(_deontic_kg.static_triples),
#             "obligations":   len(_deontic_kg.by_deontic.get("obligation", [])),
#             "prohibitions":  len(_deontic_kg.by_deontic.get("prohibition", [])),
#         }
#         return jsonify(result)

#     except Exception as e:
#         log.exception("Error in /api/check")
#         return jsonify({"error": str(e)}), 500

import threading
import uuid

# In-memory store for background jobs
JOBS = {}

@app.route("/api/check", methods=["POST"])
def check_compliance():
    if not _ready:
        return jsonify({"error": "Server not ready"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    # # Save upload to a temp file
    # with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
    #     pdf_file.save(tmp.name)
    #     tmp_path = Path(tmp.name)

    # job_id = str(uuid.uuid4())
    # JOBS[job_id] = {"status": "running", "result": None}

    # def background_task():
    #     try:
    #         from scripts.run_compliance import run as run_compliance
    #         result = run_compliance(
    #             pdf_path=tmp_path,
    #             out_path=None,
    #             use_rerank=True,
    #             use_cache=False
    #         )
    #         JOBS[job_id]["result"] = result
    #         JOBS[job_id]["status"] = "done"
    #     except Exception as e:
    #         JOBS[job_id]["status"] = "error"
    #         JOBS[job_id]["result"] = str(e)
    #     finally:
    #         tmp_path.unlink(missing_ok=True)

    # Create a temp directory to hold the original file name
    tmp_dir = tempfile.mkdtemp()
    original_name = pdf_file.filename
    tmp_path = Path(tmp_dir) / original_name
    pdf_file.save(tmp_path)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running", "result": None}

    def background_task():
        try:
            from scripts.run_compliance import run as run_compliance
            result = run_compliance(
                pdf_path=tmp_path,
                out_path=None,
                use_rerank=True,
                use_cache=True   # <-- FIX 1: Enable the cache!
            )
            JOBS[job_id]["result"] = result
            JOBS[job_id]["status"] = "done"
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["result"] = str(e)
        finally:
            # Clean up the temp directory when done
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Start the long task in a separate thread
    thread = threading.Thread(target=background_task)
    thread.start()

    # Return immediately so the browser doesn't time out
    return jsonify({"job_id": job_id}), 202

@app.route("/api/check/status/<job_id>")
def check_status(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    
    job = JOBS[job_id]
    return jsonify(job)

# ── Serve Frontend ────────────────────────────────────────────────────────────
# This allows Flask to serve the frontend directly, avoiding CORS/Codespaces issues.
FRONTEND_DIR = Path(__file__).parent.parent.parent / "docs"

@app.route('/')
def serve_index():
    return send_file(FRONTEND_DIR / "index.html")

@app.route('/<path:path>')
def serve_static(path):
    file_path = FRONTEND_DIR / path
    if file_path.exists():
        return send_file(file_path)
    return "Not found", 404

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info("Starting PolicyRAG API on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
