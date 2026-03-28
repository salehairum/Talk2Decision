import json
import os
import sys
from importlib import import_module, util
from pathlib import Path
from typing import Any, Dict, List

from config import LLMConfig, load_config


DEFAULT_NO_DECISION = {
	"decision": "No clear decision found",
	"confidence": "Low",
	"evidence": [],
}


def _bootstrap_env() -> None:
	"""Load .env from common project locations before reading config."""
	try:
		dotenv_loader = None
		try:
			dotenv_module = import_module("dotenv")
			dotenv_loader = getattr(dotenv_module, "load_dotenv", None)
		except Exception:
			dotenv_loader = None

		current_dir = Path(__file__).resolve().parent
		project_root = Path(__file__).resolve().parents[2]
		workspace_root = Path(__file__).resolve().parents[3]

		# Try explicit paths first so env loading does not depend on process cwd.
		candidates = [
			current_dir / ".env",
			current_dir.parent / ".env",
			project_root / ".env",
			workspace_root / ".env",
			Path.cwd() / ".env",
		]

		def _parse_env_file(env_path: Path) -> None:
			for raw_line in env_path.read_text(encoding="utf-8").splitlines():
				line = raw_line.strip()
				if not line or line.startswith("#"):
					continue
				if line.startswith("export "):
					line = line[7:].strip()
				if "=" not in line:
					continue
				key, value = line.split("=", 1)
				key = key.strip()
				value = value.strip()
				if (
					len(value) >= 2
					and value[0] == value[-1]
					and value[0] in {'"', "'"}
				):
					value = value[1:-1]
				if key:
					os.environ[key] = value

		seen = set()
		for env_path in candidates:
			resolved = str(env_path.resolve())
			if resolved in seen:
				continue
			seen.add(resolved)
			if env_path.exists():
				if callable(dotenv_loader):
					dotenv_loader(dotenv_path=env_path, override=True)
				else:
					_parse_env_file(env_path)
				break

		if callable(dotenv_loader):
			# Keep default behavior as a final fallback.
			dotenv_loader(override=False)
	except Exception:
		# Environment variables can still be provided by the shell.
		return


def _load_context_entrypoint() -> Any:
	"""Load context-extraction entrypoint from sibling backend folder."""
	backend_dir = Path(__file__).resolve().parents[1]
	context_main = backend_dir / "context-extraction" / "pipeline" / "main.py"
	if not context_main.exists():
		raise FileNotFoundError(f"Context entrypoint not found: {context_main}")

	context_dir = str(context_main.parent)
	if context_dir not in sys.path:
		sys.path.insert(0, context_dir)

	spec = util.spec_from_file_location("context_pipeline_main", context_main)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module spec from: {context_main}")

	module = util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return getattr(module, "entrypoint")


def retrieve_chunks(slack_export_path: str, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
	"""Retrieve top relevant chunks/messages from context extraction pipeline."""
	entrypoint = _load_context_entrypoint()
	results = entrypoint(slack_export_path=slack_export_path, top_k=top_k, query=query)
	if not isinstance(results, list):
		return []
	return results


def chunks_to_messages(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
	"""Map retrieval chunks into LLM input schema: user/text/timestamp."""
	messages: List[Dict[str, str]] = []
	seen = set()

	for chunk in chunks:
		if not isinstance(chunk, dict):
			continue
		msg = chunk.get("message", {})
		if not isinstance(msg, dict):
			continue

		user = str(msg.get("author_name") or msg.get("author_id") or "Unknown")
		timestamp = str(msg.get("timestamp") or "")
		text = ""

		window = chunk.get("window")
		if isinstance(window, dict) and window.get("text"):
			text = str(window.get("text"))
		else:
			text = str(msg.get("content") or msg.get("content_clean") or "")

		text = text.strip()
		if not text:
			continue

		key = (user, text, timestamp)
		if key in seen:
			continue
		seen.add(key)
		messages.append({"user": user, "text": text, "timestamp": timestamp})

	return messages


def format_messages(messages: List[Dict[str, Any]]) -> str:
	"""Format Slack-style messages into a readable, deterministic block."""
	lines: List[str] = []
	for idx, msg in enumerate(messages, start=1):
		user = str(msg.get("user", "")).strip()
		text = str(msg.get("text", "")).strip()
		timestamp = str(msg.get("timestamp", "")).strip()
		lines.append(f"[{idx}] user={user} | timestamp={timestamp}\\n{text}")
	return "\\n\\n".join(lines)


def build_prompt(messages_block: str, query: str) -> List[Any]:
	system_prompt = (
		"You are an AI system that analyzes team conversations to extract decisions and supporting evidence."
	)

	user_prompt = (
		"You are given:\n"
		"1. A list of messages (each contains user, text, timestamp)\n"
		"2. A user query\n\n"
		"Your task:\n"
		"1. Identify the final decision related to the query\n"
		"2. Find the exact message where the decision is clearly stated\n"
		"3. Identify who made the decision and when\n"
		"4. Provide supporting evidence messages\n\n"
		"STRICT RULES:\n"
		"- ONLY use the provided messages\n"
		"- DO NOT hallucinate or infer missing data\n"
		"- The quoted decision message MUST match the original text exactly\n"
		"- If no clear decision is found, say so explicitly\n\n"
		"OUTPUT FORMAT:\n\n"
		"First, provide a natural language answer:\n\n"
		'\"Final Decision: <decision in plain English>.\n\n'
		"This decision was made by <user> at <timestamp>, as stated in the message:\n"
		'\"<exact message text>\"\n\n'
		"Explanation:\n"
		"Briefly explain how the conversation led to this decision.\"\n\n"
		"Then provide structured JSON:\n\n"
		"{\n"
		'  \"decision\": \"string\",\n'
		'  \"decision_made_by\": \"string\",\n'
		'  \"timestamp\": \"string\",\n'
		'  \"decision_message\": \"exact message text\",\n'
		'  \"confidence\": \"High/Medium/Low\",\n'
		'  \"evidence\": [\n'
		"    {\n"
		'      \"user\": \"string\",\n'
		'      \"text\": \"exact message text\",\n'
		'      \"timestamp\": \"string\"\n'
		"    }\n"
		"  ]\n"
		"}\n\n"
		"If no decision is found, return:\n\n"
		'\"Final Decision: No clear decision found.\"\n\n'
		"{\n"
		'  \"decision\": \"No clear decision found\",\n'
		'  \"decision_made_by\": null,\n'
		'  \"timestamp\": null,\n'
		'  \"decision_message\": null,\n'
		'  \"confidence\": \"Low\",\n'
		'  \"evidence\": []\n'
		"}\n\n"
		"Now analyze the following messages:\n\n"
		f"{messages_block}\n\n"
		"User Query:\n"
		f"{query}"
	)

	# Tuple-style messages keep the implementation simple and LangChain-compatible.
	return [("system", system_prompt), ("human", user_prompt)]


def get_chat_llm(config: LLMConfig) -> Any:
	if not config.api_key:
		raise ValueError("Missing API key for selected provider.")

	provider = config.provider.lower()

	if provider == "openai":
		chat_module = import_module("langchain_openai")
		chat_cls = getattr(chat_module, "ChatOpenAI")
		kwargs: Dict[str, Any] = {
			"model": config.model_name,
			"api_key": config.api_key,
			"temperature": config.temperature,
			"max_tokens": config.max_tokens,
		}
		if config.api_base:
			kwargs["base_url"] = config.api_base
		return chat_cls(**kwargs)

	if provider == "groq":
		chat_module = import_module("langchain_groq")
		chat_cls = getattr(chat_module, "ChatGroq")
		kwargs = {
			"model": config.model_name,
			"api_key": config.api_key,
			"temperature": config.temperature,
			"max_tokens": config.max_tokens,
		}
		if config.api_base:
			kwargs["base_url"] = config.api_base
		return chat_cls(**kwargs)

	if provider == "gemini":
		chat_module = import_module("langchain_google_genai")
		chat_cls = getattr(chat_module, "ChatGoogleGenerativeAI")
		kwargs = {
			"model": config.model_name,
			"google_api_key": config.api_key,
			"temperature": config.temperature,
			"max_output_tokens": config.max_tokens,
		}
		return chat_cls(**kwargs)

	raise ValueError(f"Unsupported LLM provider: {config.provider}")


def safe_json_parse(raw: str) -> Dict[str, Any]:
	"""Parse JSON robustly, including simple fenced responses."""
	text = raw.strip()

	if text.startswith("```"):
		text = text.strip("`")
		if text.lower().startswith("json"):
			text = text[4:].strip()

	try:
		data = json.loads(text)
		if isinstance(data, dict):
			return data
	except json.JSONDecodeError:
		pass

	start = text.find("{")
	end = text.rfind("}")
	if start != -1 and end != -1 and end > start:
		snippet = text[start : end + 1]
		try:
			data = json.loads(snippet)
			if isinstance(data, dict):
				return data
		except json.JSONDecodeError:
			pass

	raise ValueError("LLM response is not valid JSON.")


def normalize_output(data: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Enforce schema and remove evidence that does not exactly match input messages."""
	decision = str(data.get("decision", "")).strip()
	confidence = str(data.get("confidence", "Low")).strip()
	evidence = data.get("evidence", [])

	if confidence not in {"High", "Medium", "Low"}:
		confidence = "Low"

	if not isinstance(evidence, list):
		evidence = []

	allowed = {
		(str(m.get("user", "")), str(m.get("text", "")), str(m.get("timestamp", "")))
		for m in messages
	}

	cleaned_evidence: List[Dict[str, str]] = []
	for item in evidence:
		if not isinstance(item, dict):
			continue
		user = str(item.get("user", ""))
		text = str(item.get("text", ""))
		timestamp = str(item.get("timestamp", ""))
		if (user, text, timestamp) in allowed:
			cleaned_evidence.append({"user": user, "text": text, "timestamp": timestamp})

	if not decision:
		return DEFAULT_NO_DECISION.copy()

	if decision == "No clear decision found":
		return DEFAULT_NO_DECISION.copy()

	if not cleaned_evidence:
		# Prevent unsupported claims when there is no valid source evidence.
		return DEFAULT_NO_DECISION.copy()

	return {
		"decision": decision,
		"confidence": confidence,
		"evidence": cleaned_evidence,
	}


def format_decision_response(result: Dict[str, Any]) -> str:
	"""Render a clean, user-facing decision summary with evidence."""
	decision = str(result.get("decision", "No clear decision found")).strip()
	confidence = str(result.get("confidence", "Low")).strip()
	evidence = result.get("evidence", [])

	lines: List[str] = [
		"Decision Summary",
		f"Decision: {decision}",
		f"Confidence: {confidence}",
		"",
		"Evidence:",
	]

	if not isinstance(evidence, list) or not evidence:
		lines.append("- No supporting evidence found.")
		return "\n".join(lines)

	for idx, item in enumerate(evidence, start=1):
		if not isinstance(item, dict):
			continue
		user = str(item.get("user", "Unknown")).strip() or "Unknown"
		text = str(item.get("text", "")).strip()
		timestamp = str(item.get("timestamp", "")).strip()

		meta = f"{idx}. {user}"
		if timestamp:
			meta += f" ({timestamp})"

		lines.append(meta)
		lines.append(f"   {text}")

	return "\n".join(lines)


def extract_decision(
	messages: List[Dict[str, Any]],
	query: str,
	config: LLMConfig,
) -> Dict[str, Any]:
	"""Run extraction end-to-end with graceful fallbacks."""
	if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
		raise ValueError("messages must be a list of dict objects.")

	try:
		messages_block = format_messages(messages)
		prompt_messages = build_prompt(messages_block, query)
		print("=============================")
		print("Prompt Messages: ", prompt_messages)

		llm = get_chat_llm(config)
		response = llm.invoke(prompt_messages)
		print("=============================")
		print("LLM Raw Response: ", response)
		content = response.content if isinstance(response.content, str) else json.dumps(response.content)
		print("=============================")
		print("LLM Content to Parse: ", content)

		parsed = safe_json_parse(content)
		return normalize_output(parsed, messages)
	except Exception as e:
		# Print error details before fallback so user can see what failed.
		print("=============================")
		print(f"ERROR in extract_decision: {type(e).__name__}: {e}")
		import traceback
		traceback.print_exc()
		print("=============================")
		# Graceful fallback while preserving the strict required schema.
		return DEFAULT_NO_DECISION.copy()


def main() -> None:
	if len(sys.argv) < 2:
		print("Usage: python llm_pipeline.py <slack_export.json> [top_k]")
		return

	slack_export_path = sys.argv[1]
	try:
		top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
	except ValueError:
		top_k = 8

	_bootstrap_env()
	config = load_config()
	print("Decision pipeline ready. Type a query (type 'exit' to quit).")

	while True:
		query = input("Query > ").strip()
		if query.lower() in {"exit", "quit"}:
			print("Exiting.")
			break
		if not query:
			continue

		try:
			chunks = retrieve_chunks(slack_export_path=slack_export_path, query=query, top_k=top_k)
			print("Chunks: ", chunks)
		except Exception as exc:
			print(f"Retrieval error: {exc}")
			continue

		messages = chunks_to_messages(chunks)
		print("============================")
		print("Messages: ", messages)
		result = extract_decision(messages, query, config)
		print("============================")
		print("Raw Result: ", result)
		print(format_decision_response(result))
		print("============================")
		print("Ready for next query.")


if __name__ == "__main__":
	main()
