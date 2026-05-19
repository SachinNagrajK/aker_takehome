"""OpenAI provider."""
from __future__ import annotations

from .base import LLMProvider, ProviderUnavailable
from ..config import get_settings


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
