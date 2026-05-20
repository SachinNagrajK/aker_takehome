"""Compile the v2 LangGraph state machine (post-rewrite).

Topology:

    START → extract_scope ──(conflict|missing)──▶ clarify (interrupt)
                                                      │
                                                      └── (re-route) ──▶ extract_scope
            └──(single|compare)──▶ seed_messages ──▶ agent ◀────┐
                                                       │         │
                                                  (tool_calls?)   │
                                                       ├─yes─▶ tools ┘
                                                       └─no ─▶ compose ─▶ END

State persistence: `InMemorySaver` keyed by `thread_id = conversation_id`,
which lets `interrupt()` pause cleanly and `Command(resume=...)` continue
the same run on the next /chat call.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from .state import ChatState
from .nodes import (
    MAX_TURNS_DEFAULT,
    agent,
    agent_should_continue,
    clarify,
    compose,
    extract_scope,
    scope_router,
    seed_messages,
    tools,
)


def _build():
    g = StateGraph(ChatState)

    g.add_node("extract_scope", extract_scope)
    g.add_node("clarify", clarify)
    g.add_node("seed_messages", seed_messages)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("compose", compose)

    g.add_edge(START, "extract_scope")
    g.add_conditional_edges(
        "extract_scope",
        scope_router,
        {"clarify": "clarify", "seed_messages": "seed_messages"},
    )
    # After the user answers the clarification, re-dispatch through the same
    # router so an invalid reply ("garbage") loops back to clarify, and a
    # valid one proceeds to seed_messages.
    g.add_conditional_edges(
        "clarify",
        scope_router,
        {"clarify": "clarify", "seed_messages": "seed_messages"},
    )

    g.add_edge("seed_messages", "agent")
    g.add_conditional_edges(
        "agent",
        agent_should_continue,
        {"tools": "tools", "compose": "compose"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("compose", END)

    return g.compile(checkpointer=InMemorySaver())


@lru_cache(maxsize=1)
def get_graph():
    return _build()


def run_chat(
    *,
    property_code: str | None,
    user_message: str,
    llm_provider: str,
    model: str,
    conversation_id: str,
    resume_value: str | list[str] | None = None,
    max_turns: int = MAX_TURNS_DEFAULT,
) -> dict[str, Any]:
    """Run (or resume) one chat turn.

    Returns the final state plus a `clarification` field when the graph
    paused at the clarify node.
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    if resume_value is not None:
        out = graph.invoke(Command(resume=resume_value), config=config)
    else:
        initial: ChatState = {
            "user_message": user_message,
            "dropdown_property_code": property_code,
            "llm_provider": llm_provider,
            "model": model,
            "conversation_id": conversation_id,
            "tool_history": [],
            "turn_count": 0,
            "max_turns": max_turns,
            "messages": [],
        }
        out = graph.invoke(initial, config=config)

    snapshot = graph.get_state(config)
    interrupts = getattr(snapshot, "interrupts", None) or []
    if interrupts:
        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        return {
            "clarification": payload,
            "scope": snapshot.values.get("scope"),
            "conversation_id": conversation_id,
            "paused": True,
        }

    out["clarification"] = None
    out["conversation_id"] = conversation_id
    out["paused"] = False
    return out
