"""Anthropic provider."""
from __future__ import annotations

from .base import LLMProvider, ProviderUnavailable
from ..config import get_settings


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        # Anthropic uses a separate `system` field, not a system message.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]

        # For json mode we lean on a prompt directive — Anthropic supports
        # structured outputs but a directive is portable and usually enough
        # for the small JSON-pick prompts we use.
        if response_format == "json_object" and system_parts:
            system_parts.append(
                "Respond with a single JSON object only. No prose, no markdown fences."
            )

        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens or 1024,
            temperature=temperature,
            system="\n\n".join(system_parts) if system_parts else None,
            messages=[{"role": m["role"], "content": m["content"]} for m in chat],
        )
        # Concatenate text blocks.
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)
