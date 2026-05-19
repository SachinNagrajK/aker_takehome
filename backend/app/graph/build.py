"""Wire the nodes into a LangGraph state machine."""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import StateGraph, END

from .state import ChatState
from .nodes import (
    validate_scope,
    classify_query,
    sql_node,
    rag_node,
    hybrid_node,
    compose_response,
    route_selector,
)


def _build() -> Any:  # noqa: F821 — typing.Any only needed at runtime
    g = StateGraph(ChatState)

    g.add_node("validate_scope", validate_scope)
    g.add_node("classify_query", classify_query)
    g.add_node("sql_node", sql_node)
    g.add_node("rag_node", rag_node)
    g.add_node("hybrid_node", hybrid_node)
    g.add_node("compose_response", compose_response)

    g.set_entry_point("validate_scope")
    g.add_edge("validate_scope", "classify_query")

    g.add_conditional_edges(
        "classify_query",
        route_selector,
        {
            "sql":    "sql_node",
            "rag":    "rag_node",
            "hybrid": "hybrid_node",
        },
    )

    g.add_edge("sql_node",    "compose_response")
    g.add_edge("rag_node",    "compose_response")
    g.add_edge("hybrid_node", "compose_response")
    g.add_edge("compose_response", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Return the compiled graph (one-shot, cached)."""
    return _build()


def run_chat(
    *,
    property_code: str,
    user_message: str,
    llm_provider: str,
    model: str,
) -> dict:
    """Convenience entry-point used by the FastAPI /chat route."""
    graph = get_graph()
    initial: ChatState = {
        "property_code": property_code,
        "user_message": user_message,
        "llm_provider": llm_provider,
        "model": model,
    }
    return graph.invoke(initial)
