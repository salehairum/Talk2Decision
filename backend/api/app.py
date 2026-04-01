"""
Flask API backend for Talk2Decision.

Handles:
- File upload and storage
- Processing status tracking
- Query execution on processed files
"""

import os
import sys
import json
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add sibling modules to path
backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_dir / "context-extraction" / "pipeline"))
sys.path.insert(0, str(backend_dir / "llm-pipeline"))

from loader import load_slack_export
from preprocess import preprocess_messages
from filter_for_search import SentenceTransformerEmbedder, build_index
from query import run_query
from config import load_config, get_config_options, SUPPORTED_PROVIDERS
from llm_pipeline import extract_decision, chunks_to_messages, format_decision_response


app = Flask(__name__, static_folder=str(backend_dir.parents[0] / "frontend"), static_url_path="/static")
CORS(app)

DATA_DIR = backend_dir / "context-extraction" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = backend_dir.parents[0] / "frontend"

ALLOWED_EXTENSIONS = {"json"}

# In-memory cache: {file_id: {"status": "...", "index": ..., "messages": ...}}
processing_cache = {}
processing_lock = threading.Lock()


def _get_top_retrieval_score(chunks: list[dict]) -> float:
    """Return the highest numeric retrieval score from query chunks."""
    scores: list[float] = []
    for item in chunks:
        if not isinstance(item, dict):
            continue
        value = item.get("score")
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    return max(scores)


def _iso_now() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    """Terminal-friendly logger for real-time backend visibility."""
    print(message, flush=True)


def _update_processing_state(file_id: str, **updates) -> None:
    """Atomically update per-file processing state with heartbeat metadata."""
    now = _iso_now()
    with processing_lock:
        state = processing_cache.setdefault(
            file_id,
            {
                "status": "starting",
                "progress": 0,
                "message": "Initializing...",
                "error": None,
                "created_at": now,
                "updated_at": now,
                "step_started_at": now,
            },
        )

        if "status" in updates and updates["status"] != state.get("status"):
            updates.setdefault("step_started_at", now)

        state.update(updates)
        state["updated_at"] = now


def get_file_id(filename: str) -> str:
    """Generate a unique file ID from filename."""
    return Path(filename).stem


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_file_async(filepath: str, file_id: str) -> None:
    """Load, preprocess, and build index for a file (runs in background)."""
    _log(f"[PROCESS] Starting async processing for {file_id} from {filepath}")
    try:
        _update_processing_state(
            file_id,
            status="starting",
            progress=0,
            message="Initializing processing pipeline...",
            error=None,
            created_at=_iso_now(),
        )
        _log(f"[PROCESS] Cache entry created for {file_id}")

        # Step 1: Load
        _log(f"[PROCESS] Step 1: Loading file {filepath}")
        _update_processing_state(
            file_id,
            status="loading",
            progress=20,
            message="Loading Slack export...",
        )
        raw_messages = load_slack_export(filepath)
        _log(f"[PROCESS] Step 1 complete: loaded {len(raw_messages)} messages")

        # Step 2: Preprocess
        _log(f"[PROCESS] Step 2: Preprocessing {len(raw_messages)} messages")
        _update_processing_state(
            file_id,
            status="preprocessing",
            progress=40,
            message=f"Preprocessing {len(raw_messages)} messages...",
        )
        clean_messages = preprocess_messages(raw_messages)
        _log(f"[PROCESS] Step 2 complete: preprocessed to {len(clean_messages)} messages")

        # Step 3: Build index
        _log(f"[PROCESS] Step 3: Building index for {len(clean_messages)} messages")
        _update_processing_state(
            file_id,
            status="indexing",
            progress=60,
            message=(
                "Building embeddings and index (first run can take several minutes while models download)..."
            ),
        )

        _log("[PROCESS] Initializing sentence-transformer embedder")
        embedder = SentenceTransformerEmbedder()
        _update_processing_state(
            file_id,
            status="indexing",
            progress=70,
            message="Embedding model loaded. Building vector database...",
        )
        index = build_index(clean_messages, embedder=embedder)
        _log(f"[PROCESS] Step 3 complete: index built")

        # Step 4: Done
        _log(f"[PROCESS] Step 4: Marking {file_id} as completed")
        _update_processing_state(
            file_id,
            status="completed",
            progress=100,
            message="Ready for queries",
            index=index,
            messages=clean_messages,
            embedder=embedder,
            uploaded_at=_iso_now(),
        )
        _log(f"[PROCESS] Async processing COMPLETED for {file_id}")
    except Exception as exc:
        _log(f"[PROCESS] ERROR in async processing for {file_id}: {exc}")
        import traceback
        traceback.print_exc()
        _update_processing_state(
            file_id,
            status="failed",
            progress=0,
            message=f"Error: {str(exc)}",
            error=str(exc),
        )


@app.route("/", methods=["GET"])
def index():
    """Serve the frontend HTML."""
    if (FRONTEND_DIR / "index.html").exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({"error": "Frontend not found"}), 404


@app.route("/styles.css", methods=["GET"])
def frontend_styles():
    """Serve main frontend stylesheet for the root HTML page."""
    stylesheet = FRONTEND_DIR / "styles.css"
    if stylesheet.exists():
        return send_from_directory(str(FRONTEND_DIR), "styles.css")
    return jsonify({"error": "Stylesheet not found"}), 404


@app.route("/style.css", methods=["GET"])
def frontend_style_alias():
    """Backward-compatible alias in case HTML points to style.css."""
    stylesheet = FRONTEND_DIR / "style.css"
    if stylesheet.exists():
        return send_from_directory(str(FRONTEND_DIR), "style.css")

    fallback = FRONTEND_DIR / "styles.css"
    if fallback.exists():
        return send_from_directory(str(FRONTEND_DIR), "styles.css")

    return jsonify({"error": "Stylesheet not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/llm/options", methods=["GET"])
def llm_options():
    """Return selectable provider/model options for clients."""
    return jsonify(get_config_options()), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload a JSON Slack export file."""
    if "file" not in request.files:
        _log("[ERROR] No file in request")
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        _log("[ERROR] Empty filename")
        return jsonify({"error": "Empty filename"}), 400

    if not is_allowed_file(file.filename):
        _log(f"[ERROR] File type not allowed: {file.filename}")
        return jsonify({"error": "Only .json files allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = DATA_DIR / filename
    file.save(filepath)
    _log(f"[UPLOAD] File saved: {filepath}")

    file_id = get_file_id(filename)
    _log(f"[UPLOAD] Generated file_id: {file_id}")

    # Start async processing
    thread = threading.Thread(target=process_file_async, args=(str(filepath), file_id), daemon=True)
    thread.start()
    _log(f"[UPLOAD] Processing thread started for {file_id}")

    return (
        jsonify(
            {
                "file_id": file_id,
                "filename": filename,
                "status": "queued",
                "message": "File uploaded, processing started",
            }
        ),
        202,
    )


@app.route("/status/<file_id>", methods=["GET"])
def get_status(file_id: str):
    """Get processing status for a file."""
    with processing_lock:
        if file_id not in processing_cache:
            return jsonify({"error": "File not found"}), 404

        cache = dict(processing_cache[file_id])

    now = datetime.utcnow()
    updated_at = cache.get("updated_at")
    step_started_at = cache.get("step_started_at")

    heartbeat_seconds = None
    step_elapsed_seconds = None
    try:
        if isinstance(updated_at, str):
            heartbeat_seconds = max(0, int((now - datetime.fromisoformat(updated_at)).total_seconds()))
        if isinstance(step_started_at, str):
            step_elapsed_seconds = max(0, int((now - datetime.fromisoformat(step_started_at)).total_seconds()))
    except ValueError:
        # Keep diagnostics optional if date parsing fails.
        pass

    message = cache.get("message", "")
    if cache.get("status") == "indexing" and isinstance(step_elapsed_seconds, int) and step_elapsed_seconds > 90:
        message = (
            f"{message} Still running for {step_elapsed_seconds}s. "
            "If this is the first run, model download can take a few minutes."
        )

    # Return safe subset for client
    return jsonify(
        {
            "file_id": file_id,
            "status": cache.get("status"),
            "progress": cache.get("progress", 0),
            "message": message,
            "error": cache.get("error"),
            "uploaded_at": cache.get("uploaded_at"),
            "created_at": cache.get("created_at"),
            "updated_at": cache.get("updated_at"),
            "step_started_at": cache.get("step_started_at"),
            "heartbeat_seconds": heartbeat_seconds,
            "step_elapsed_seconds": step_elapsed_seconds,
        }
    ), 200


@app.route("/files", methods=["GET"])
def list_files():
    """List all processed files."""
    with processing_lock:
        files = []
        for file_id, cache in processing_cache.items():
            if cache.get("status") == "completed":
                files.append(
                    {
                        "file_id": file_id,
                        "status": "ready",
                        "uploaded_at": cache.get("uploaded_at"),
                    }
                )
        return jsonify({"files": files}), 200


@app.route("/query", methods=["POST"])
def query_file():
    """Run a decision extraction query on a processed file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    file_id = data.get("file_id")
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 8)
    llm_provider = str(data.get("llm_provider", "")).strip().lower()
    llm_model = str(data.get("llm_model", "")).strip()

    if not file_id or not query:
        return jsonify({"error": "Missing file_id or query"}), 400

    if llm_provider and llm_provider not in SUPPORTED_PROVIDERS:
        return (
            jsonify(
                {
                    "error": (
                        "Unsupported llm_provider. "
                        f"Allowed values: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
                    )
                }
            ),
            400,
        )

    with processing_lock:
        if file_id not in processing_cache:
            return jsonify({"error": "File not found"}), 404

        cache = processing_cache[file_id]
        if cache.get("status") != "completed":
            return jsonify({"error": "File not ready for queries"}), 400

        index = cache.get("index")
        embedder = cache.get("embedder")

    try:
        _log(f"[QUERY] file_id={file_id} query={query!r} top_k={top_k}")

        # Run retrieval
        results = run_query(index, query=query, top_k=top_k, embedder=embedder)
        _log(f"[QUERY] Retrieved {len(results)} chunks")
        
        # Convert chunks to message format
        messages = chunks_to_messages(results)
        
        # Extract decision
        config = load_config(
            provider_override=llm_provider or None,
            model_override=llm_model or None,
        )
        _log(
            f"[QUERY] Using LLM provider={config.provider} model={config.model_name}"
        )
        decision = extract_decision(messages, query, config)

        retrieval_score = round(_get_top_retrieval_score(results), 6)
        if isinstance(decision, dict):
            decision["confidence_score"] = retrieval_score
            decision["confidence"] = retrieval_score

        decision_response = format_decision_response(decision)
        _log("[QUERY] LLM formatted response:\n" + decision_response)
        
        return jsonify(
            {
                "file_id": file_id,
                "query": query,
                "chunks_retrieved": len(results),
                "llm": {
                    "provider": config.provider,
                    "model": config.model_name,
                },
                "decision": decision,
                "decision_response": decision_response,
            }
        ), 200
    except Exception as exc:
        _log(f"[QUERY] ERROR: {exc}")
        return jsonify({"error": f"Query failed: {str(exc)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
