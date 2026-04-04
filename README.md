# Talk2Decision

Minimalistic web app to extract decisions from Slack chat exports using AI.

## What Is Implemented

### Architecture

- **Frontend** (`frontend/index.html`): Single-page web app for file upload and querying
- **Backend API** (`backend/api/app.py`): Flask REST API for file management and querying
- **Context Extraction** (`backend/context-extraction/`): Message loading, preprocessing, semantic search
- **LLM Pipeline** (`backend/llm-pipeline/`): Decision extraction with multi-provider LLM support

### Features

- **File Management**: Upload and storage in `backend/context-extraction/data/`
- **Real-time Processing**: Progress tracking from upload → indexing → ready
- **Semantic Search**: Message embedding and indexing (SentenceTransformers + Chroma)
- **Hybrid Search**: Keyword + semantic search over messages
- **Multi-Provider LLM**: OpenAI, Groq, Gemini support
- **Decision Extraction**: Strict JSON extraction with evidence and action items
- **Hallucination Prevention**: Exact-match evidence validation

#### Cross-File Decision Deduplication ✨

- **Same Topic Tracking**: Same query across multiple files/days → single decision
- **Confidence-Aware Updates**: High-confidence decisions protected from low-confidence overwrites
- **Source File Tracking**: All contributing files tracked in `source_files` array
- **Multiple Topics**: Different queries tracked independently simultaneously
- **Decision History**: Complete audit trail of all updates with timestamps
- **Smart Merging**: Evidence and action items only updated when quality improves

## Project Structure

```
Talk2Decision/
├── frontend/
│   └── index.html                 # Single-page web UI
├── backend/
│   ├── api/
│   │   └── app.py                 # Flask REST API
│   ├── llm-pipeline/
│   │   ├── llm_pipeline.py        # Decision extraction logic
│   │   └── config.py              # LLM configuration
│   └── context-extraction/
│       ├── pipeline/
│       │   ├── main.py            # Entrypoint and query execution
│       │   ├── loader.py          # Slack export loader
│       │   ├── preprocess.py      # Message cleaning
│       │   ├── filter_for_search.py # Semantic/keyword search
│       │   └── query.py           # Query execution
│       └── data/                  # Uploaded JSON chat files
├── requirements.txt
└── README.md
```

## Output Schema

Success case:

## Configuration

The backend
{
	"decision": "string",
	"confidence": "High/Medium/Low",
	"evidence": [
		{
			"user": "string",
			"text": "string",
			"timestamp": "string"
		}
	]
}
```

No-decision fallback:

```json
{
	"decision": "No clear decision found",
	"confidence": "Low",
	"evidence": []
}
```

The code reads these environment variables:

- `LLM_PROVIDER` (`openai`, `groq`, or `gemini`)
- `LLM_API_KEY` (required to call the LLM)
- `OPENAI_API_KEY` (optional provider-specific key)
- `GROQ_API_KEY` (optional provider-specific key)
- `GOOGLE_API_KEY` (optional provider-specific key)
- `LLM_MODEL_NAME` (default: `gpt-4o-mini`)
- `LLM_TEMPERATURE` (default: `0`)
- `LLM_MAX_TOKENS` (default: `500`)
- `LLM_API_BASE` (optional for custom-compatible endpoints)

Notes:
- If `LLM_PROVIDER` is not set, the default is `openai`.
- You can also use a prefixed model name like `groq/llama-3.1-8b-instant`; provider is inferred and the prefix is stripped automatically.
The `.env` file (if present) is loaded automatically by the backend

Because `python-dotenv` is included, a `.env` file (if present) is loaded automatically.

## Install

```bash
pip install -r requirements.txt
```

## Example .env (Optional)

```env
LLM_PROVIDER=openai
LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=gpt-4o-mini
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=500
# Optional
# LLM_API_BASE=https://your-compatible-endpoint/v1
```

Groq example:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL_NAME=llama-3.1-8b-instant
```

Gemini example:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
LLM_MODEL_NAME=gemini-1.5-flash
```

## Run

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
# Copy and edit .env or set environment variables
export LLM_PROVIDER=groq
export GROQ_API_KEY=your_key_here
```

### Backend (Flask API)

From the repository root:

```bash
python backend/api/app.py
```

This starts the API server at `http://127.0.0.1:5000`.

The API exposes:
- `POST /upload` – Upload a Slack export JSON file
- `GET /status/<file_id>` – Check processing progress
- `GET /files` – List processed files
- `POST /query` – Query a processed file for decision extraction

### Frontend (Web UI)

Once the backend is running, open your browser and navigate to:

```
http://127.0.0.1:5000/
```

The frontend will:
1. Load the web interface from the backend
2. Connect to the API at `http://127.0.0.1:5000`
3. Allow you to upload JSON Slack export files
4. Show real-time processing progress
5. Display uploaded files in a sidebar
6. Let you query each file and view extracted decisions

### Workflow

1. Start the backend API:
   ```bash
   python backend/api/app.py
   ```
   You should see:
   ```
   * Running on http://127.0.0.1:5000
   ```

2. Open your browser and go to `http://127.0.0.1:5000/`

3. Upload a Slack export JSON file (e.g., from `backend/context-extraction/data/`)

4. Wait for processing to complete (loading → preprocessing → indexing → ready)

5. Once ready, the file appears in the sidebar

6. Enter a natural-language query about the chat

7. View the extracted decision with supporting evidence from the messages

### Troubleshooting

**404 Not Found on `http://127.0.0.1:5000/`:**
- Make sure the backend is running with `python backend/api/app.py`
- Make sure `frontend/index.html` exists in the correct location
- Refresh your browser

**File upload fails:**
- Make sure `backend/context-extraction/data/` directory exists (created automatically)
- Ensure the JSON file is valid Slack export format

**Processing hangs:**
- Check the terminal running Flask for error messages
- Large files with many messages may take time during indexing

## Cross-File Decision Deduplication

The system intelligently tracks decisions across multiple Slack exports, preventing duplicate decisions for the same topic when files are from different days.

### How It Works

#### Query Matching
- Queries are normalized (lowercase, trimmed) to match naturally
- Same topic across different files → same decision is updated
- Different topics → separate decisions tracked independently

#### Confidence Hierarchy
```
High (confidence rank 3) > Medium (rank 2) > Low (rank 1)
```

A new extraction will **only update** an existing decision if:
```
new_confidence >= existing_confidence
```

This prevents low-quality extractions from overwriting high-quality decisions.

#### Source File Tracking
All files that contributed to a decision are logged:
```json
{
  "decision_id": 1,
  "query": "What is the decision?",
  "decision": "Go with the blue theme",
  "confidence": "High",
  "source_files": ["2026-03-23.json", "2026-03-26.json", "2026-04-03.json"],
  "status": "In-Progress"
}
```

### Behavior Examples

**Scenario 1: Same Topic, Different Days**
```
Day 1: Upload 2026-03-23.json
       Query: "What is the decision?"
       Result: HIGH confidence → Decision #1 created

Day 2: Upload 2026-03-26.json
       Same query, MEDIUM confidence
       Result: Decision #1 updated (Medium ≥ baseline)
               source_files: ["2026-03-23", "2026-03-26"]

Day 3: Upload 2026-04-03.json
       Same query, LOW confidence
       Result: Decision #1 NOT updated (Low < High)
               source_files: ["2026-03-23", "2026-03-26", "2026-04-03"]
               Status moved to "In-Progress" (re-exported)
```

**Scenario 2: Multiple Topics**
```
Query 1: "What is the decision?"        → Decision #1 (tracked)
Query 2: "What are the action items?"   → Decision #2 (separate)
Query 3: "What were the main topics?"   → Decision #3 (separate)
```

Each topic is tracked independently.

### Decision History

Every update to a decision is logged with timestamps:
```
Decision #1 Update History:
├─ SOURCE_FILE: "2026-03-23" → "2026-03-26"
│  (@ 2026-04-04 10:38:31)
├─ DECISION: "light theme with accents" → "Go with blue theme"
│  (@ 2026-04-04 10:38:31)
└─ CONFIDENCE: "Medium" → "High"
   (@ 2026-04-04 10:39:15)
```

### Evidence and Action Items

Evidence and action items are only updated when confidence improves:
- **High → Medium/Low**: Keep existing (don't downgrade quality)
- **Medium → High**: Update with better information
- **Low → Any**: Update (any improvement welcomed)
- **Always**: Track source file, move status to "In-Progress"

### API: Decision Query Endpoint

```bash
curl -X POST http://127.0.0.1:5000/query \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "2026-03-23",
    "query": "What is the decision?",
    "top_k": 8
  }'
```

Response:
```json
{
  "decision_id": 1,
  "action": "created",
  "decision": {
    "decision": "Go with the blue theme",
    "confidence": "High",
    "evidence": [...],
    "action_items": [...]
  }
}
```

- `"action": "created"` – New decision was created
- `"action": "updated"` – Existing decision was updated

## Technical Details

### Database Schema

**decisions** table includes:
- `id`: Unique decision identifier
- `query`: Normalized query text (matched across files)
- `extracted_decision`: The decision text
- `confidence`: High/Medium/Low
- `source_files`: JSON array of contributing file IDs
- `status`: Open → In-Progress → Resolved
- `created_at`, `updated_at`: Timestamps
- Relationships to evidence, action items, and history

**decision_history** table tracks:
- `field_name`: Which field changed (decision, confidence, source_file, etc.)
- `old_value` → `new_value`: What changed
- `changed_at`: When the change happened

### Deduplication Logic

```python
# 1. Normalize query
normalized_query = query.strip().lower()

# 2. Lookup existing decision
result = db.execute(
  "SELECT id FROM decisions WHERE LOWER(TRIM(query)) = LOWER(?)",
  (normalized_query,)
)

# 3. Compare confidence
if existing:
  if new_confidence >= existing_confidence:
    update_decision()  # Update text, evidence, items
  else:
    keep_existing()    # Preserve high-quality decision

# 4. Always track source file
add_to_source_files(file_id)
move_to_in_progress()
```

### Fresh Database Connections

The deduplication lookup uses fresh database connections to ensure:
- No cached/stale data from ORM session
- Always sees latest committed state
- Guarantees consistent cross-file matching