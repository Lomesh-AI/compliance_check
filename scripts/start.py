"""Prepare runtime data and start the production WSGI server."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.json"
FAISS_PATH = ROOT / "data" / "index" / "faiss.index"


def main() -> None:
    if not CHUNKS_PATH.exists() or not FAISS_PATH.exists():
        print("Index artifacts are incomplete; building data/index artifacts...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "src.ingestion.ingest"],
            cwd=ROOT,
            check=True,
        )

    missing = [str(path) for path in (CHUNKS_PATH, FAISS_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Index build completed without required files: {', '.join(missing)}")

    os.execvp(
        "gunicorn",
        [
            "gunicorn",
            "--bind",
            "0.0.0.0:8080",
            "--workers",
            "1",
            "--timeout",
            "120",
            "--access-logfile",
            "-",
            "--error-logfile",
            "-",
            "src.api.app:app",
        ],
    )


if __name__ == "__main__":
    main()