# Talk2Decision

MVP Python pipeline to extract final decisions from Slack-style messages using an LLM.

## What Is Implemented

- Input format: list of message objects plus a user query.
- Config-driven model setup from environment variables.
- Prompt rules to force strict JSON output and reduce hallucinations.
- Evidence validation against exact input messages.
- Safe JSON parsing with graceful fallback.
- Example `main()` with dummy messages and sample query.

Pipeline file:
- `backend/llm-pipeline/llm_pipeline.py`

Config file:
- `backend/llm-pipeline/config.py`

## Output Schema

Success case:

```json
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

## Do I Need config.yaml or .env?

- `config.yaml`: Not required.
- `.env`: Optional but recommended for local development.

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
- `LLM_API_KEY` works as a generic override for any provider.

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

From the repository root:

```bash
python backend/llm-pipeline/llm_pipeline.py
```

On Windows with an absolute Python path:

```powershell
& "C:/Program Files/Python313/python.exe" "backend/llm-pipeline/llm_pipeline.py"
```