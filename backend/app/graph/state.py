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

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _tool_history_reducer(left: list, right: list) -> list:
    """Custom reducer with explicit-reset semantics.

    Standard `operator.add` would accumulate `tool_history` across turns of
    the same conversation, leaking turn-1 steps into turn-2's UI trace.
    With this reducer:

      - `enter_turn` returns `tool_history: []` at the start of every turn
        to CLEAR the previous turn's steps.
      - `tools` returns the new step(s) it ran, which append to the now-empty
        list.

    The only node that should return an empty list is `enter_turn`. The
    executor (`tools`) always returns a non-empty list when it appends.
    """
    if right == []:
        return []
    return (left or []) + (right or [])


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
    dropdown_property_code: Any  # str | list[str] | None — single or multi-select
    llm_provider: str
    model: str
    conversation_id: str | None

    # --- resolved scope ---
    scope: dict[str, Any]
    property_name: str | None
    # Time-scope dimension (v5). Mirrors `scope` but for "as of which month":
    #   {kind: "latest"|"specific"|"missing"|"any", month: "2025-04-01"|None, label: str|None}
    time_scope: dict[str, Any]

    # --- loop control ---
    turn_count: int             # number of agent turns taken
    max_turns: int              # hard cap (default 8)

    # --- clarification flow (LangGraph interrupt) ---
    clarification: dict[str, Any] | None   # set by clarify_router

    # --- parallel trace for UI (full tool results, not truncated).
    # Uses a custom reducer so each turn starts with an empty list. ---
    tool_history: Annotated[list[ToolStep], _tool_history_reducer]

    # --- final outputs (filled by compose) ---
    answer_markdown: str
    components: list[dict[str, Any]]
    sources: list[dict[str, str]]
    route: str                  # "sql" | "rag" | "hybrid" | "agent"
    gave_up: bool
