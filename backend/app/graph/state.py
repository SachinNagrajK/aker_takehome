"""LangGraph state schema.

Every node reads from and writes to this TypedDict. `property_code` is the
load-bearing field — present at the entry point and read by every node.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


Route = Literal["sql", "rag", "hybrid"]


class ChatState(TypedDict, total=False):
    # --- inputs ---
    property_code: str
    property_name: str
    user_message: str
    llm_provider: str
    model: str

    # --- routing decision ---
    route: Route
    sql_tool: str | None        # e.g. "get_rent_trend"
    sql_args: dict[str, Any]    # kwargs for the picked tool

    # --- tool outputs ---
    sql_result: dict[str, Any] | None
    rag_chunks: list[dict[str, Any]]
    rag_sources: list[dict[str, str]]

    # --- final outputs ---
    answer_markdown: str
    components: list[dict[str, Any]]
    sources: list[dict[str, str]]
