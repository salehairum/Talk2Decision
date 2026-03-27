import json
from importlib import import_module
from typing import Any, Dict, List

from config import LLMConfig, load_config


DEFAULT_NO_DECISION = {
	"decision": "No clear decision found",
	"confidence": "Low",
	"evidence": [],
}


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
		"You are an information extraction system. "
		"Use ONLY the provided conversation messages. "
		"Do NOT infer, assume, or add details that are not explicitly present. "
		"If there is no clear decision, return exactly: "
		'{"decision":"No clear decision found","confidence":"Low","evidence":[]}. '
		"Return STRICT JSON only with keys: decision, confidence, evidence. "
		"confidence must be one of: High, Medium, Low. "
		"evidence must be an array of objects with keys: user, text, timestamp. "
		"Each evidence.text MUST match a provided message text exactly. "
		"No markdown, no extra keys, no commentary."
	)

	user_prompt = (
		"User query:\\n"
		f"{query}\\n\\n"
		"Conversation messages:\\n"
		f"{messages_block}\\n\\n"
		"Task:\\n"
		"1) Determine whether a final decision was made.\\n"
		"2) Extract the final decision text.\\n"
		"3) Extract exact evidence messages from the conversation only.\\n"
		"4) If unclear, return the no-decision JSON."
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

		llm = get_chat_llm(config)
		response = llm.invoke(prompt_messages)
		content = response.content if isinstance(response.content, str) else json.dumps(response.content)

		parsed = safe_json_parse(content)
		return normalize_output(parsed, messages)
	except Exception:
		# Graceful fallback while preserving the strict required schema.
		return DEFAULT_NO_DECISION.copy()


def main() -> None:
	messages = [
        {"user": "Ali", "text": "We need to finalize the frontend theme today", "timestamp": "1"},
        {"user": "Sara", "text": "Yeah current one feels too plain", "timestamp": "2"},
        {"user": "Usman", "text": "I was thinking maybe dark mode as default?", "timestamp": "3"},
        {"user": "Hina", "text": "Dark mode is trendy but not always readable", "timestamp": "4"},
        {"user": "Ali", "text": "What about light theme with blue accents?", "timestamp": "5"},
        {"user": "Sara", "text": "Blue looks professional tbh", "timestamp": "6"},
        {"user": "Usman", "text": "We can also consider purple, looks modern", "timestamp": "7"},
        {"user": "Hina", "text": "Purple might be too flashy for our use case", "timestamp": "8"},
        {"user": "Ali", "text": "Agree, we want something clean and simple", "timestamp": "9"},
        {"user": "Sara", "text": "Blue + white combo is safe and clean", "timestamp": "10"},
        {"user": "Usman", "text": "Okay I am convinced, blue works", "timestamp": "11"},
        {"user": "Ali", "text": "So final decision: light theme with blue accents", "timestamp": "12"},
        {"user": "Hina", "text": "Yes let's lock that", "timestamp": "13"},
        {"user": "Sara", "text": "Done 👍", "timestamp": "14"}
	]
	query = "What was the final decision made about theme?"

	config = load_config()
	result = extract_decision(messages, query, config)
	print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
	main()
