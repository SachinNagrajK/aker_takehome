"""LLM registry — names + availability surfaced to the frontend.

This is the only LLM-related module that survived v3. The legacy
`app/llm/` package (custom `LLMProvider` base + 3 hand-rolled adapters)
was removed because the v2 agent uses LangChain chat classes
(`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`) which already
implement `.bind_tools()` correctly. See `graph/nodes.py:_make_llm` for
the runtime dispatcher.
"""
from __future__ import annotations

from .config import MODELS, available_providers


class ProviderUnavailable(RuntimeError):
    """Raised when an LLM provider is selected but its API key isn't set,
    or when an unknown provider/model name is supplied."""


def validate_model(provider: str, model: str) -> str:
    """Return `model` if (provider, model) is in the registry, else raise."""
    if provider not in MODELS:
        raise ProviderUnavailable(f"Unknown provider: {provider!r}")
    if model not in MODELS[provider]:
        raise ProviderUnavailable(
            f"Unknown model {model!r} for provider {provider!r}. "
            f"Valid: {MODELS[provider]}"
        )
    return model


def list_llms() -> list[dict]:
    """Manifest for `GET /llms` — provider, models, availability."""
    avail = available_providers()
    return [
        {"provider": p, "models": MODELS[p], "available": avail.get(p, False)}
        for p in MODELS
    ]
