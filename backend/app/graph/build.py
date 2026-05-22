"""Compile the v2 LangGraph state machine (post-rewrite).

Topology:

    START → extract_scope ──(conflict|missing)──▶ clarify (interrupt)
                                                      │
                                                      └── (re-route) ──▶ extract_scope
            └──(single|compare)──▶ enter_turn ──▶ agent ◀────┐
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
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, AIMessageChunk
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
    enter_turn,
    tools,
)


# Human-readable progress lines, surfaced as `step` events on the SSE stream
# so the UI can show "Pulling property summary…" instead of a static spinner.
_TOOL_REASONING = {
    "get_property_summary":   "Pulling property summary",
    "get_unit_mix":           "Breaking down unit mix",
    "get_occupancy":          "Checking occupancy",
    "get_rent_trend":         "Loading rent trend",
    "get_expiring_leases":    "Scanning expiring leases",
    "get_top_balances":       "Ranking outstanding balances",
    "get_unit_charges":       "Itemising charges for unit",
    "compare_units":          "Comparing units side by side",
    "compare_properties":     "Comparing properties",
    "list_units":             "Filtering units",
    "execute_scoped_sql":     "Running a custom SQL query",
    "search_property_pages":  "Searching marketing pages & photos",
    "render_chart":           "Rendering a chart",
}


def _reasoning_for(tool_name: str, args: dict | None = None) -> str:
    line = _TOOL_REASONING.get(tool_name, f"Calling {tool_name}")
    if args and isinstance(args, dict):
        if "property_code" in args:
            line += f" · {args['property_code']}"
        if "query" in args and isinstance(args["query"], str):
            q = args["query"].strip()
            if q:
                line += f" · “{q[:40]}{'…' if len(q) > 40 else ''}”"
    return line


def _build():
    g = StateGraph(ChatState)

    g.add_node("extract_scope", extract_scope)
    g.add_node("clarify", clarify)
    g.add_node("enter_turn", enter_turn)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("compose", compose)

    g.add_edge(START, "extract_scope")
    g.add_conditional_edges(
        "extract_scope",
        scope_router,
        {"clarify": "clarify", "enter_turn": "enter_turn"},
    )
    # After the user answers the clarification, re-dispatch through the same
    # router so an invalid reply ("garbage") loops back to clarify, and a
    # valid one proceeds to enter_turn.
    g.add_conditional_edges(
        "clarify",
        scope_router,
        {"clarify": "clarify", "enter_turn": "enter_turn"},
    )

    g.add_edge("enter_turn", "agent")
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
    property_code,  # str | list[str] | None — single or multi-select dropdown
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


# ---------------------------------------------------------------------------
# Streaming variant — yields {type, ...} events suitable for SSE
# ---------------------------------------------------------------------------

async def run_chat_stream(
    *,
    property_code,
    user_message: str,
    llm_provider: str,
    model: str,
    conversation_id: str,
    resume_value: str | list[str] | None = None,
    max_turns: int = MAX_TURNS_DEFAULT,
) -> AsyncIterator[dict[str, Any]]:
    """Async iterator over progress events for one chat turn.

    Event shapes (`type` field discriminator):
      - step    : {type, node, message}                       — node entered
      - tool    : {type, tool, args, reasoning}               — tool started
      - tool_end: {type, tool, ok, duration_ms}               — tool finished
      - delta   : {type, text}                                — streaming token from agent LLM
      - clarification: {type, payload}                         — paused at interrupt
      - done    : {type, response: <full ChatResponse dict>}  — final state
      - error   : {type, message}                              — fatal error
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": conversation_id}}

    if resume_value is not None:
        graph_input = Command(resume=resume_value)
    else:
        graph_input: Any = {
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

    # Track tools that started but haven't been flushed yet (for tool_end timing).
    seen_tool_call_ids: set[str] = set()
    streamed_so_far = ""  # last AI text we've already emitted as deltas

    try:
        async for stream_mode, payload in graph.astream(
            graph_input,
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if stream_mode == "updates":
                # `payload` is {node_name: state_update_dict_or_list}.
                if not isinstance(payload, dict):
                    continue
                for node_name, update in payload.items():
                    if node_name == "extract_scope":
                        yield {"type": "step", "node": node_name,
                               "message": "Resolving property scope"}
                    elif node_name == "enter_turn":
                        yield {"type": "step", "node": node_name,
                               "message": "Reading your question"}
                    elif node_name == "agent":
                        # Agent ran a turn — check if it queued any tool calls.
                        msgs = (update or {}).get("messages") if isinstance(update, dict) else None
                        if isinstance(msgs, list):
                            for m in msgs:
                                tool_calls = getattr(m, "tool_calls", None) or []
                                for tc in tool_calls:
                                    tcid = tc.get("id") or ""
                                    if tcid in seen_tool_call_ids:
                                        continue
                                    seen_tool_call_ids.add(tcid)
                                    name = tc.get("name") or ""
                                    args = tc.get("args") or {}
                                    yield {
                                        "type": "tool",
                                        "tool": name,
                                        "args": args,
                                        "reasoning": _reasoning_for(name, args),
                                    }
                    elif node_name == "tools":
                        # Tools node finished — surface per-tool result.
                        steps = (update or {}).get("tool_history") if isinstance(update, dict) else None
                        if isinstance(steps, list):
                            for step in steps:
                                yield {
                                    "type": "tool_end",
                                    "tool": step.get("tool"),
                                    "ok": bool(step.get("ok")),
                                    "duration_ms": step.get("duration_ms"),
                                    "error": step.get("error"),
                                }
                    elif node_name == "compose":
                        yield {"type": "step", "node": node_name,
                               "message": "Composing answer"}
                    elif node_name == "clarify":
                        yield {"type": "step", "node": node_name,
                               "message": "Need clarification"}
            elif stream_mode == "messages":
                # `payload` is (BaseMessage_or_chunk, metadata_dict).
                try:
                    msg_chunk, metadata = payload
                except (TypeError, ValueError):
                    continue
                node = (metadata or {}).get("langgraph_node")
                # Stream tokens from the AGENT's natural-language reply (the
                # one that ends a turn with no tool_calls). Compose-node
                # tokens are streamed too when the agent gave up.
                if node not in {"agent", "compose"}:
                    continue
                if not isinstance(msg_chunk, (AIMessage, AIMessageChunk)):
                    continue
                text = getattr(msg_chunk, "content", None)
                if not isinstance(text, str) or not text:
                    continue
                # Tool-calling messages carry empty/tool-arg content; we
                # only want to forward real prose. Skip if the chunk also
                # has tool_call_chunks (it's mid tool-call assembly).
                if getattr(msg_chunk, "tool_call_chunks", None):
                    continue
                if getattr(msg_chunk, "tool_calls", None):
                    continue
                # Emit the new portion only (defensive in case chunks aren't
                # strictly incremental for some providers).
                if text.startswith(streamed_so_far):
                    delta = text[len(streamed_so_far):]
                    streamed_so_far = text
                else:
                    delta = text
                    streamed_so_far = text
                if delta:
                    yield {"type": "delta", "text": delta}
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return

    # Stream complete — build the final ChatResponse-equivalent dict.
    snapshot = graph.get_state(config)
    interrupts = getattr(snapshot, "interrupts", None) or []
    if interrupts:
        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        yield {
            "type": "clarification",
            "payload": payload,
            "scope": snapshot.values.get("scope"),
        }
        return

    state = snapshot.values
    scope = state.get("scope") or {}
    yield {
        "type": "done",
        "response": {
            "property_code": scope.get("code"),
            "property_codes": scope.get("codes") or [],
            "scope_kind": scope.get("kind", "single"),
            "scope_source": scope.get("source"),
            "scope_enforced": True,
            "answer_markdown": state.get("answer_markdown", ""),
            "components": state.get("components", []) or [],
            "sources": state.get("sources", []) or [],
            "tool_trace": [
                {
                    "tool": s.get("tool"),
                    "args": s.get("args") or {},
                    "ok": bool(s.get("ok")),
                    "error": s.get("error"),
                    "duration_ms": s.get("duration_ms"),
                }
                for s in (state.get("tool_history") or [])
            ],
            "route": state.get("route", "agent"),
            "gave_up": bool(state.get("gave_up")),
            "llm": {"provider": llm_provider, "model": model},
            "conversation_id": conversation_id,
        },
    }
