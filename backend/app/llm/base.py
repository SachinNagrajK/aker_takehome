"""LLM provider abstraction.

The graph and the routes only know about `LLMProvider` — concrete adapters
live in sibling modules. Each provider has the same surface:

    provider.generate(messages, model, temperature=0.2,
                      response_format=None, max_tokens=None) -> str

`messages` is the OpenAI-style list:
    [{"role": "system", "content": "..."},
     {"role": "user",   "content": "..."}]

Anthropic and Gemini adapters translate to/from their native shapes
internally so callers don't care.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        response_format: str | None = None,   # "json_object" | None
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant's text reply (or JSON string)."""


class ProviderUnavailable(RuntimeError):
    """Raised when a provider is selected but no API key is configured."""
