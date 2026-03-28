# Talk2Decision

Minimalistic web app to extract decisions from Slack chat exports using AI.

## What Is Implemented

### Architecture

- **Frontend** (`frontend/index.html`): Single-page web app for file upload and querying
- **Backend API** (`backend/api/app.py`): Flask REST API for file management and querying
- **Context Extraction** (`backend/context-extraction/`): Message loading, preprocessing, semantic search
- **LLM Pipeline** (`backend/llm-pipeline/`): Decision extraction with multi-provider LLM support

### Features

- File upload and storage in `backend/context-extraction/data/`
- Real-time processing progress tracking
- Message embedding and indexing (SentenceTransformers + Chroma)
- Hybrid keyword + semantic search over messages
- Multi-provider LLM support (OpenAI, Groq, Gemini)
- Strict JSON decision extraction with evidence
- Hallucination prevention via exact-match evidence validation

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