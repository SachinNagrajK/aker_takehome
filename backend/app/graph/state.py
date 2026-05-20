"""LangGraph state schema (v2 rewrite).

This is the canonical MessagesState pattern: the LLM conversation is a
single list of messages with the `add_messages` reducer. The agent node
appends `AIMessage`s (some carrying `tool_calls`); the tools node appends
`ToolMessage`s with results.

`tool_history` is kept as a *parallel* collection for the UI's Tool Trace
panel — it stores full tool results that we deliberately truncate when
serialised into `ToolMessage.content` to keep prompt budgets sane.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ToolStep(TypedDict, total=False):
    tool: str
    args: dict[str, Any]
    ok: bool
    result: Any                 # full result, dict-typed for the UI
    error: str | None
    duration_ms: int | None
    tool_call_id: str | None


class ChatState(TypedDict, total=False):
    # --- conversation (canonical) ---
    messages: Annotated[list[BaseMessage], add_messages]

    # --- inputs / context ---
    user_message: str
    dropdown_property_code: str | None
    llm_provider: str
    model: str
    conversation_id: str | None

    # --- resolved scope ---
    scope: dict[str, Any]
    property_name: str | None

    # --- loop control ---
    turn_count: int             # number of agent turns taken
    max_turns: int              # hard cap (default 8)

    # --- clarification flow (LangGraph interrupt) ---
    clarification: dict[str, Any] | None   # set by clarify_router

    # --- parallel trace for UI (full tool results, not truncated) ---
    tool_history: Annotated[list[ToolStep], add]

    # --- final outputs (filled by compose) ---
    answer_markdown: str
    components: list[dict[str, Any]]
    sources: list[dict[str, str]]
    route: str                  # "sql" | "rag" | "hybrid" | "agent"
    gave_up: bool
