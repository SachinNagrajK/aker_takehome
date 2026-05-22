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
    # v2: property_code is OPTIONAL — scope may also come from the message
    # text or from a clarification follow-up. v3: accepts a single code OR
    # a list of codes (multi-select dropdown → compare mode from the start).
    property_code: str | list[str] | None = None
    message: str = Field(min_length=1)
    llm_provider: str = "openai"
    model: str = "gpt-4o-mini"
    # Optional — frontend supplies the same id on follow-up clarification
    # replies so the graph can resume via the LangGraph checkpointer.
    conversation_id: str | None = None
    # When set, this is the user's reply to a clarification interrupt.
    # Either a single property code or a comma-separated list.
    clarification_reply: str | None = None


class Source(BaseModel):
    label: str
    url: str | None = None


class UIComponent(BaseModel):
    type: Literal[
        "kpi", "table",
        "bar_chart", "line_chart", "comparison_chart",
        "pie_chart", "donut_chart",
        "image",
    ]
    title: str
    # Free-form payload validated per type by the renderer.
    data: dict[str, Any] = Field(default_factory=dict)


class Clarification(BaseModel):
    question: str
    options: list[str] = Field(default_factory=list)
    scope_kind: str | None = None      # "conflict" | "missing"


class ToolTraceStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    error: str | None = None
    duration_ms: int | None = None


class ChatResponse(BaseModel):
    # The resolved scope shown to the user (single code, list, or None when
    # we paused at a clarification interrupt).
    property_code: str | None = None
    property_codes: list[str] = Field(default_factory=list)
    scope_kind: str = "single"          # "single" | "compare" | "conflict" | "missing"
    scope_source: str | None = None     # "query" | "dropdown" | "resumed"
    scope_enforced: bool = True

    answer_markdown: str = ""
    components: list[UIComponent] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_trace: list[ToolTraceStep] = Field(default_factory=list)
    route: str = "agent"
    llm: dict[str, str]
    gave_up: bool = False

    # When set, the graph paused for clarification. The frontend should
    # render this and POST the user's choice back with the same
    # conversation_id + a `clarification_reply` field.
    clarification: Clarification | None = None
    conversation_id: str | None = None
