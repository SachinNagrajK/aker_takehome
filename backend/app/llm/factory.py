"""LLM provider factory — picks a concrete adapter by name.

This is the only place that knows about all three concrete providers, so the
graph nodes stay provider-agnostic.
"""
from __future__ import annotations

from .base import LLMProvider, ProviderUnavailable
from ..config import MODELS, available_providers


def get_provider(name: str) -> LLMProvider:
    """Return a provider instance or raise ProviderUnavailable."""
    n = (name or "").strip().lower()
    if n == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    if n == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if n == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider()
    raise ProviderUnavailable(f"Unknown LLM provider: {name!r}")


def validate_model(provider: str, model: str) -> str:
    """Ensure (provider, model) is in the registry; return the model name."""
    if provider not in MODELS:
        raise ProviderUnavailable(f"Unknown provider: {provider!r}")
    if model not in MODELS[provider]:
        raise ProviderUnavailable(
            f"Unknown model {model!r} for provider {provider!r}. "
            f"Valid: {MODELS[provider]}"
        )
    return model


def list_llms() -> list[dict]:
    """Manifest for GET /llms — provider, models, availability."""
    avail = available_providers()
    return [
        {"provider": p, "models": MODELS[p], "available": avail.get(p, False)}
        for p in MODELS
    ]
