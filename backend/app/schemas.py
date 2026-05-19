"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class PropertyOut(BaseModel):
    property_code: str
    property_name: str
    property_type: str


class LLMOption(BaseModel):
    provider: str
    models: list[str]
    available: bool


class ChatRequest(BaseModel):
    property_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    llm_provider: str = "openai"
    model: str = "gpt-4o-mini"


class Source(BaseModel):
    label: str
    url: str | None = None


class UIComponent(BaseModel):
    type: Literal["kpi", "table", "bar_chart", "line_chart"]
    title: str
    # Free-form payload validated per type by the renderer.
    data: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    property_code: str
    scope_enforced: bool = True
    answer_markdown: str
    components: list[UIComponent] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    route: str  # "sql" | "rag" | "hybrid"
    llm: dict[str, str]
