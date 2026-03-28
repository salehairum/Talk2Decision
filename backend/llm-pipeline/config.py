import os
from importlib import import_module
from dataclasses import dataclass
from typing import Optional


def _clean_env_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {'"', "'"}
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def load_dotenv() -> None:
    try:
        dotenv_module = import_module("dotenv")
        dotenv_loader = getattr(dotenv_module, "load_dotenv", None)
        if callable(dotenv_loader):
            dotenv_loader()
    except Exception:
        # Environment variables can still be set by the shell.
        return None


# Load .env if present; environment variables still work normally.
load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 500
    api_base: Optional[str] = None


def load_config() -> LLMConfig:
    """Load config from environment variables with sane defaults.

    Set environment variables as needed:
    - LLM_PROVIDER (openai/groq/gemini)
    - LLM_MODEL_NAME
    - LLM_API_KEY (generic override for any provider)
    - OPENAI_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY
    - LLM_TEMPERATURE
    - LLM_MAX_TOKENS
    - LLM_API_BASE
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini").strip()

    # Allow model names like "groq/llama-3.1-8b-instant" and infer provider.
    if "/" in model_name:
        maybe_provider, maybe_model = model_name.split("/", 1)
        if maybe_provider.lower() in {"openai", "groq", "gemini"}:
            if not provider:
                provider = maybe_provider.lower()
            model_name = maybe_model.strip()

    if not provider:
        if model_name.startswith("gemini"):
            provider = "gemini"
        else:
            provider = "openai"
    generic_api_key = _clean_env_value(os.getenv("LLM_API_KEY"))
    openai_api_key = _clean_env_value(os.getenv("OPENAI_API_KEY"))
    groq_api_key = _clean_env_value(os.getenv("GROQ_API_KEY"))
    google_api_key = _clean_env_value(os.getenv("GOOGLE_API_KEY"))

    if provider == "openai":
        api_key = openai_api_key or generic_api_key
    elif provider == "groq":
        api_key = groq_api_key or generic_api_key
    elif provider == "gemini":
        api_key = google_api_key or generic_api_key
    else:
        # Keep behavior predictable with unsupported values.
        provider = "openai"
        api_key = openai_api_key or generic_api_key

    temperature_raw = os.getenv("LLM_TEMPERATURE", "0")
    max_tokens_raw = os.getenv("LLM_MAX_TOKENS", "500")
    api_base = os.getenv("LLM_API_BASE")

    try:
        temperature = float(temperature_raw)
    except ValueError:
        temperature = 0.0

    try:
        max_tokens = int(max_tokens_raw)
    except ValueError:
        max_tokens = 500

    return LLMConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        api_base=api_base,
    )
