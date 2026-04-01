import os
from importlib import import_module
from dataclasses import dataclass
from typing import Any, Dict, Optional


SUPPORTED_PROVIDERS = {"groq", "gemini"}

PROVIDER_DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "gemini": "gemini-2.5-flash",
}

PROVIDER_MODEL_ENV = {
    "groq": "GROQ_MODEL",
    "gemini": "GEMINI_MODEL",
}


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
    provider: str = "groq"
    model_name: str = "llama-3.1-8b-instant"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 500
    api_base: Optional[str] = None


def _resolve_provider(raw_provider: Optional[str], raw_model_name: str) -> str:
    provider = (raw_provider or "").strip().lower()
    model_name = (raw_model_name or "").strip()

    if "/" in model_name:
        maybe_provider, _ = model_name.split("/", 1)
        if maybe_provider.lower() in SUPPORTED_PROVIDERS and not provider:
            provider = maybe_provider.lower()

    if provider in SUPPORTED_PROVIDERS:
        return provider

    if model_name.startswith("gemini"):
        return "gemini"

    return "groq"


def _resolve_model_name(provider: str, model_override: Optional[str]) -> str:
    explicit_model = _clean_env_value(model_override)
    if explicit_model:
        return explicit_model

    llm_model_name = _clean_env_value(os.getenv("LLM_MODEL_NAME"))
    if llm_model_name:
        if "/" in llm_model_name:
            maybe_provider, maybe_model = llm_model_name.split("/", 1)
            if maybe_provider.lower() in SUPPORTED_PROVIDERS and maybe_model.strip():
                return maybe_model.strip()
        return llm_model_name

    provider_model_env = PROVIDER_MODEL_ENV.get(provider)
    provider_model = _clean_env_value(os.getenv(provider_model_env or ""))
    if provider_model:
        return provider_model

    return PROVIDER_DEFAULT_MODELS.get(provider, PROVIDER_DEFAULT_MODELS["groq"])


def get_available_models() -> Dict[str, str]:
    """Return configured default model for each provider."""
    return {
        provider: _resolve_model_name(provider, None)
        for provider in sorted(SUPPORTED_PROVIDERS)
    }


def load_config(provider_override: Optional[str] = None, model_override: Optional[str] = None) -> LLMConfig:
    """Load config from environment variables with sane defaults.

    Set environment variables as needed:
    - LLM_PROVIDER (groq/gemini)
    - LLM_MODEL_NAME (global model override)
    - GROQ_MODEL / GEMINI_MODEL (provider-specific defaults)
    - LLM_API_KEY (generic override for any provider)
    - GROQ_API_KEY / GEMINI_API_KEY
    - LLM_TEMPERATURE
    - LLM_MAX_TOKENS
    - LLM_API_BASE
    """
    env_provider = _clean_env_value(os.getenv("LLM_PROVIDER"))
    raw_provider = provider_override if provider_override is not None else env_provider
    model_name = _resolve_model_name(
        _resolve_provider(raw_provider, model_override or ""),
        model_override,
    )
    provider = _resolve_provider(raw_provider, model_name)

    generic_api_key = _clean_env_value(os.getenv("LLM_API_KEY"))
    groq_api_key = _clean_env_value(os.getenv("GROQ_API_KEY"))
    gemini_api_key = _clean_env_value(os.getenv("GEMINI_API_KEY"))
    google_api_key = _clean_env_value(os.getenv("GOOGLE_API_KEY"))

    if provider == "groq":
        api_key = groq_api_key or generic_api_key
    elif provider == "gemini":
        api_key = gemini_api_key or google_api_key or generic_api_key
    else:
        # Keep behavior predictable with unsupported values.
        provider = "groq"
        api_key = groq_api_key or generic_api_key
        model_name = _resolve_model_name(provider, model_override)

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


def get_config_options() -> Dict[str, Any]:
    """Expose provider/model options for UI clients."""
    active = load_config()
    available_models = get_available_models()
    return {
        "active_provider": active.provider,
        "active_model": active.model_name,
        "providers": [
            {
                "name": provider,
                "default_model": available_models.get(provider, ""),
            }
            for provider in sorted(SUPPORTED_PROVIDERS)
        ],
    }
