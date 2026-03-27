import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


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
    generic_api_key = os.getenv("LLM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if provider == "openai":
        api_key = generic_api_key or openai_api_key
    elif provider == "groq":
        api_key = generic_api_key or groq_api_key
    elif provider == "gemini":
        api_key = generic_api_key or google_api_key
    else:
        # Keep behavior predictable with unsupported values.
        provider = "openai"
        api_key = generic_api_key or openai_api_key

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
