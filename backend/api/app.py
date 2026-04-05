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
sys.path.insert(0, str(backend_dir))  # Add backend dir for models import

from loader import load_slack_export
from preprocess import preprocess_messages
from filter_for_search import SentenceTransformerEmbedder, build_index
from query import run_query
from config import load_config, get_config_options, SUPPORTED_PROVIDERS
from llm_pipeline import extract_decision, chunks_to_messages, format_decision_response

# Import database models
from models import db, Decision, DecisionEvidence, ActionItem, DecisionHistory, Stakeholder


app = Flask(__name__, static_folder=str(backend_dir.parents[0] / "frontend"), static_url_path="/static")
CORS(app)

# Initialize database
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{backend_dir}/talk2decision.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    db.create_all()

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
        
        # Check if this decision already exists
        # Match by: QUERY ALONE (across all files/days)
        # Normalize query for comparison (lowercase, strip whitespace)
        from sqlalchemy import select, func, text
        
        # Normalize the current query for consistent matching
        normalized_query = query.strip().lower()
        _log(f"[QUERY] Looking for existing decision matching: {normalized_query!r}")
        
        # Try to find existing decision using raw SQL with fresh connection
        existing_decision = None
        try:
            # Get a fresh connection from the pool
            with db.engine.connect() as conn:
                # Use raw SQL for guaranteed fresh view of the database
                result = conn.execute(
                    text("SELECT id FROM decisions WHERE LOWER(TRIM(query)) = LOWER(:query) LIMIT 1"),
                    {"query": normalized_query}
                ).scalar()
                
                if result:
                    existing_id = result
                    _log(f"[QUERY] Found existing decision ID={existing_id} in database")
                    # Now load it in the main session
                    existing_decision = db.session.get(Decision, existing_id)
                    _log(f"[QUERY] Loaded in session: {existing_decision is not None}")
                else:
                    _log(f"[QUERY] No existing decision found for: {normalized_query!r}")
        except Exception as e:
            _log(f"[QUERY] Error looking up existing decision: {e}")
            import traceback
            traceback.print_exc()
        
        # Initialize flag to track if we should update the decision
        should_update_decision = False
        
        if existing_decision:
            _log(f"[QUERY] ✓ Found existing decision (ID: {existing_decision.id})")
            decision_obj = existing_decision
            
            # Track source files: maintain list of all files that have updated this decision
            import json
            try:
                source_files = json.loads(decision_obj.source_files) if decision_obj.source_files else []
            except:
                source_files = []
            
            # Confidence levels: High > Medium > Low
            confidence_rank = {"High": 3, "Medium": 2, "Low": 1}
            old_confidence = decision_obj.confidence
            new_confidence = decision.get("confidence", "Low")
            old_rank = confidence_rank.get(old_confidence, 0)
            new_rank = confidence_rank.get(new_confidence, 0)
            
            # Check if we should update the decision
            should_update_decision = new_rank >= old_rank
            
            if not should_update_decision:
                _log(f"[QUERY] ⚠️  New confidence '{new_confidence}' is lower than existing '{old_confidence}' - NOT updating decision text")
                _log(f"[QUERY] Will still track source file and mark as In-Progress")
            
            # Always track the source file, regardless of whether we update the decision
            if file_id not in source_files:
                source_files.append(file_id)
                decision_obj.source_files = json.dumps(source_files)
                _log(f"[QUERY] Added new source file: {file_id}, total sources: {source_files}")
                
                # Log source file change in history
                history = DecisionHistory(
                    decision_id=decision_obj.id,
                    field_name="source_file",
                    old_value=",".join(source_files[:-1]) if len(source_files) > 1 else "(initial)",
                    new_value=",".join(source_files),
                    changed_by="system"
                )
                db.session.add(history)
            
            # Only update decision text and confidence if new one is better quality
            if should_update_decision:
                _log(f"[QUERY] Updating decision: confidence {old_confidence} → {new_confidence}")
                
                # Track changes in history for all updated fields
                old_decision_text = decision_obj.extracted_decision
                new_decision_text = decision.get("decision", "").strip()
                if old_decision_text != new_decision_text:
                    history = DecisionHistory(
                        decision_id=decision_obj.id,
                        field_name="decision",
                        old_value=old_decision_text,
                        new_value=new_decision_text,
                        changed_by="system"
                    )
                    db.session.add(history)
                    _log(f"[QUERY] Decision text updated")
                
                if old_confidence != new_confidence:
                    history = DecisionHistory(
                        decision_id=decision_obj.id,
                        field_name="confidence",
                        old_value=old_confidence,
                        new_value=new_confidence,
                        changed_by="system"
                    )
                    db.session.add(history)
                
                # Update decision fields with higher-confidence version
                decision_obj.extracted_decision = new_decision_text
                decision_obj.confidence = new_confidence
            else:
                _log(f"[QUERY] Keeping existing decision: confidence {old_confidence} is better than {new_confidence}")
            
            # Always move to in-progress when re-exported
            decision_obj.status = "In-Progress"
            decision_obj.updated_at = datetime.utcnow()
            
            # Only update evidence and action items if we're updating the decision
            if should_update_decision:
                # Update evidence: clear old and add new
                from sqlalchemy import delete
                db.session.execute(delete(DecisionEvidence).where(DecisionEvidence.decision_id == decision_obj.id))
                
                # Update action items: keep existing ones, add new ones
                # First, remove action items that are no longer in the extracted data
                existing_tasks = {a.task for a in decision_obj.action_items}
                new_tasks = {action.get("task", "") for action in decision.get("action_items", [])}
                tasks_to_remove = existing_tasks - new_tasks
                for task in tasks_to_remove:
                    db.session.execute(delete(ActionItem).where(
                        (ActionItem.decision_id == decision_obj.id) & (ActionItem.task == task)
                    ))
                
                # Add new action items
                for action in decision.get("action_items", []):
                    task_text = action.get("task", "")
                    from sqlalchemy import select as sql_select
                    existing_action_query = sql_select(ActionItem).where(
                        (ActionItem.decision_id == decision_obj.id) & (ActionItem.task == task_text)
                    )
                    existing_action = db.session.execute(existing_action_query).scalars().first()
                    if not existing_action:
                        action_obj = ActionItem(
                            decision_id=decision_obj.id,
                            task=task_text,
                            owner=action.get("owner"),
                            due_date=action.get("due_date"),
                            status="Open"
                        )
                        db.session.add(action_obj)
                _log(f"[QUERY] Updated evidence and action items for existing decision")
            else:
                _log(f"[QUERY] Keeping existing evidence and action items (decision not updated due to lower confidence)")
        else:
            # Create new decision
            _log(f"[QUERY] ✗ No existing decision found - CREATING new decision")
            _log(f"[QUERY] Creating decision with: file_id={file_id!r}, normalized_query={normalized_query!r}")
            # Store normalized query to ensure consistent matching
            import json
            decision_obj = Decision(
                query=query.strip(),
                extracted_decision=decision.get("decision", "").strip(),
                confidence=decision.get("confidence", "Low"),
                file_id=file_id,
                source_files=json.dumps([file_id]),  # Initialize with first source file
                status="Open",
                priority="Medium",
                category="General"
            )
            db.session.add(decision_obj)
            db.session.flush()  # Get the ID without committing yet
            
            # Save action items for new decision
            for action in decision.get("action_items", []):
                action_obj = ActionItem(
                    decision_id=decision_obj.id,
                    task=action.get("task", ""),
                    owner=action.get("owner"),
                    due_date=action.get("due_date"),
                    status="Open"
                )
                db.session.add(action_obj)
            
            # Save evidence for new decision
            evidence_list = decision.get("evidence", [])
            for evid in evidence_list:
                evidence_obj = DecisionEvidence(
                    decision_id=decision_obj.id,
                    user=evid.get("user", ""),
                    text=evid.get("text", ""),
                    timestamp=evid.get("timestamp", ""),
                    source_file=file_id
                )
                db.session.add(evidence_obj)
        
        # For existing decisions that were updated, save evidence
        if existing_decision and should_update_decision:
            evidence_list = decision.get("evidence", [])
            for evid in evidence_list:
                evidence_obj = DecisionEvidence(
                    decision_id=decision_obj.id,
                    user=evid.get("user", ""),
                    text=evid.get("text", ""),
                    timestamp=evid.get("timestamp", ""),
                    source_file=file_id
                )
                db.session.add(evidence_obj)
        
        db.session.commit()
        
        action = "updated" if existing_decision else "created"
        return jsonify(
            {
                "file_id": file_id,
                "query": query,
                "chunks_retrieved": len(results),
                "decision_id": decision_obj.id,
                "decision": decision,
                "decision_response": decision_response,
                "action": action,
                "message": f"Decision {action} successfully"
            }
        ), 200
    except Exception as exc:
        _log(f"[QUERY] ERROR: {exc}")
        return jsonify({"error": f"Query failed: {str(exc)}"}), 500


# ============================================================================
# Decision Management Endpoints
# ============================================================================

@app.route("/decisions", methods=["GET"])
def list_decisions():
    """List all tracked decisions with filters."""
    from sqlalchemy import select
    
    status = request.args.get("status")
    owner = request.args.get("owner")
    category = request.args.get("category")
    
    query = select(Decision)
    
    if status:
        query = query.where(Decision.status == status)
    if owner:
        query = query.where(Decision.owner == owner)
    if category:
        query = query.where(Decision.category == category)
    
    query = query.order_by(Decision.created_at.desc())
    decisions = db.session.execute(query).scalars().all()
    
    return jsonify({
        "count": len(decisions),
        "decisions": [d.to_dict() for d in decisions]
    }), 200


@app.route("/decisions/<int:decision_id>", methods=["GET", "DELETE"])
def get_decision(decision_id: int):
    """Get a specific decision with full details or delete it."""
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    if request.method == "DELETE":
        try:
            db.session.delete(decision)
            db.session.commit()
            return jsonify({"message": "Decision deleted successfully"}), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Failed to delete decision: {str(e)}"}), 500
    
    # GET request
    return jsonify(decision.to_dict()), 200


@app.route("/decisions/<int:decision_id>/history", methods=["GET"])
def get_decision_history(decision_id: int):
    """Get audit trail of decision changes."""
    from sqlalchemy import select
    
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    query = select(DecisionHistory).where(DecisionHistory.decision_id == decision_id).order_by(DecisionHistory.changed_at.asc())
    history = db.session.execute(query).scalars().all()
    
    return jsonify({
        "decision_id": decision_id,
        "history": [h.to_dict() for h in history]
    }), 200


@app.route("/decisions/<int:decision_id>/status", methods=["POST"])
def update_decision_status(decision_id: int):
    """Update decision status and log change."""
    data = request.get_json()
    if not data or "status" not in data:
        return jsonify({"error": "Missing status"}), 400
    
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    new_status = data.get("status").strip()
    old_status = decision.status
    
    if new_status != old_status:
        decision.status = new_status
        
        # Log change
        history = DecisionHistory(
            decision_id=decision_id,
            field_name="status",
            old_value=old_status,
            new_value=new_status,
            changed_by=data.get("changed_by", "system")
        )
        db.session.add(history)
    
    db.session.commit()
    return jsonify(decision.to_dict()), 200


@app.route("/decisions/<int:decision_id>/owner", methods=["POST"])
def update_decision_owner(decision_id: int):
    """Update decision owner and log change."""
    data = request.get_json()
    if not data or "owner" not in data:
        return jsonify({"error": "Missing owner"}), 400
    
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    new_owner = data.get("owner").strip() if data.get("owner") else None
    old_owner = decision.owner
    
    if new_owner != old_owner:
        decision.owner = new_owner
        
        # Log change
        history = DecisionHistory(
            decision_id=decision_id,
            field_name="owner",
            old_value=old_owner,
            new_value=new_owner,
            changed_by=data.get("changed_by", "system")
        )
        db.session.add(history)
    
    db.session.commit()
    return jsonify(decision.to_dict()), 200


@app.route("/decisions/<int:decision_id>/metadata", methods=["POST"])
def update_decision_metadata(decision_id: int):
    """Update decision metadata (priority, category, etc)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    # Update priority if provided
    if "priority" in data:
        new_priority = data.get("priority").strip()
        old_priority = decision.priority
        if new_priority != old_priority:
            decision.priority = new_priority
            history = DecisionHistory(
                decision_id=decision_id,
                field_name="priority",
                old_value=old_priority,
                new_value=new_priority,
                changed_by=data.get("changed_by", "system")
            )
            db.session.add(history)
    
    # Update category if provided
    if "category" in data:
        new_category = data.get("category").strip()
        old_category = decision.category
        if new_category != old_category:
            decision.category = new_category
            history = DecisionHistory(
                decision_id=decision_id,
                field_name="category",
                old_value=old_category,
                new_value=new_category,
                changed_by=data.get("changed_by", "system")
            )
            db.session.add(history)
    
    db.session.commit()
    return jsonify(decision.to_dict()), 200


@app.route("/decisions/<int:decision_id>/actions", methods=["GET"])
def get_decision_actions(decision_id: int):
    """Get all action items for a decision."""
    from sqlalchemy import select
    
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    query = select(ActionItem).where(ActionItem.decision_id == decision_id)
    actions = db.session.execute(query).scalars().all()
    return jsonify({
        "decision_id": decision_id,
        "action_items": [a.to_dict() for a in actions]
    }), 200


@app.route("/decisions/<int:decision_id>/actions", methods=["POST"])
def add_decision_action(decision_id: int):
    """Add a new action item to a decision."""
    data = request.get_json()
    if not data or "task" not in data:
        return jsonify({"error": "Missing task"}), 400
    
    decision = db.session.get(Decision, decision_id)
    if not decision:
        return jsonify({"error": "Decision not found"}), 404
    
    action = ActionItem(
        decision_id=decision_id,
        task=data.get("task").strip(),
        owner=data.get("owner", "").strip() or None,
        due_date=data.get("due_date", "").strip() or None,
        status="Open"
    )
    db.session.add(action)
    db.session.commit()
    
    return jsonify(action.to_dict()), 201


@app.route("/decisions/<int:decision_id>/actions/<int:action_id>", methods=["POST"])
def update_decision_action(decision_id: int, action_id: int):
    """Update an action item status."""
    from sqlalchemy import select
    
    data = request.get_json()
    
    query = select(ActionItem).where(
        (ActionItem.id == action_id) & (ActionItem.decision_id == decision_id)
    )
    action = db.session.execute(query).scalars().first()
    if not action:
        return jsonify({"error": "Action item not found"}), 404
    
    # Update status if provided
    if "status" in data:
        action.status = data.get("status").strip()
    
    # Update owner if provided
    if "owner" in data:
        action.owner = data.get("owner").strip() or None
    
    # Update due_date if provided
    if "due_date" in data:
        action.due_date = data.get("due_date").strip() or None
    
    db.session.commit()
    return jsonify(action.to_dict()), 200



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
