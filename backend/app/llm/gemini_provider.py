"""Gemini provider."""
from __future__ import annotations

from .base import LLMProvider, ProviderUnavailable
from ..config import get_settings


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.google_api_key:
            raise ProviderUnavailable("GOOGLE_API_KEY not set")
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        self._genai = genai

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.2,
        response_format: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat = [m for m in messages if m["role"] != "system"]

        # Gemini uses {role: user|model, parts: [...]}
        gemini_history = []
        for m in chat:
            role = "user" if m["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [m["content"]]})

        generation_config: dict = {"temperature": temperature}
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        if response_format == "json_object":
            generation_config["response_mime_type"] = "application/json"

        gm = self._genai.GenerativeModel(
            model_name=model,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
            generation_config=generation_config,
        )
        resp = gm.generate_content(gemini_history)
        return resp.text or ""
