"""Graph nodes.

Pipeline:

    validate_scope
        |
        v
    classify_query  ----+----> sql_node ----+
        |               |                   |
        |               +--> rag_node ------+
        |               |                   |
        |               +--> hybrid_node --+
        v                                   v
                              compose_response  --> END

Every node reads `state["property_code"]` and passes it to the tools, which
re-enforce it via `require_scope()`. There is no node that can read or write
data for another property.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .state import ChatState
from .components import build_components
from ..guardrails.scope import validate_property_code, system_prompt
from ..tools.sql_tools import TOOLS as SQL_TOOLS, run_tool as run_sql_tool
from ..tools.rag_tools import search_property, build_context_block, has_content
from ..llm.factory import get_provider, validate_model


# ---------------------------------------------------------------------------
# 1. validate_scope
# ---------------------------------------------------------------------------

def validate_scope(state: ChatState) -> dict[str, Any]:
    """Confirm property exists and stash its display name."""
    code, name = validate_property_code(state["property_code"])
    return {"property_code": code, "property_name": name}


# ---------------------------------------------------------------------------
# 2. classify_query
# ---------------------------------------------------------------------------

# Keywords that strongly indicate structured / rent-roll questions.
_SQL_KEYWORDS = re.compile(
    r"\b(rent|rents|occupanc|lease|leases|expir|balance|balances|unit\s+mix|"
    r"unit\s+count|how many units|vacant|vacancy|trend|over\s+(the|past)\s+\w+|"
    r"month|monthly|kpi|summary|overview|portfolio|tenant|tenants|sqft|"
    r"square\s+foot|delinquent|delinquency)\b",
    re.IGNORECASE,
)

# Keywords that indicate marketing / unstructured questions.
_RAG_KEYWORDS = re.compile(
    r"\b(amenit|floor\s*plan|floorplan|pet|policy|policies|finishes?|"
    r"appliance|laundry|parking\s+spot|neighborhood|near|nearby|locat(ion|ed)|"
    r"description|features|offer|community|amenities|gym|pool|fitness)\b",
    re.IGNORECASE,
)


def _classify_rule_based(message: str, rag_available: bool) -> str:
    """Return 'sql' | 'rag' | 'hybrid'. Falls back to SQL if RAG empty."""
    has_sql = bool(_SQL_KEYWORDS.search(message))
    has_rag = bool(_RAG_KEYWORDS.search(message))

    if has_sql and has_rag:
        return "hybrid" if rag_available else "sql"
    if has_rag and rag_available:
        return "rag"
    if has_sql:
        return "sql"
    # Default: lean SQL (rent rolls are the primary data source). If user
    # asked about amenities-ish and we have RAG, prefer hybrid.
    return "hybrid" if rag_available else "sql"


def classify_query(state: ChatState) -> dict[str, Any]:
    msg = state["user_message"]
    rag_available = has_content(state["property_code"])
    route = _classify_rule_based(msg, rag_available)
    return {"route": route}


# ---------------------------------------------------------------------------
# 3. sql_node — picks one SQL tool via LLM, runs it
# ---------------------------------------------------------------------------

def _sql_tool_picker_prompt(message: str, code: str) -> list[dict[str, str]]:
    tool_lines = []
    for name, meta in SQL_TOOLS.items():
        params = ", ".join(meta.get("params", [])) or "(no args)"
        tool_lines.append(f"  - {name}({params}): {meta['description']}")
    tools_block = "\n".join(tool_lines)

    system = (
        "You are a routing classifier. Pick ONE SQL analytical tool to answer "
        f"the user's question about property '{code}'. Return STRICT JSON only:\n"
        '  {"tool": "<tool_name>", "args": {<kwargs>}}\n\n'
        "Available tools:\n" + tools_block + "\n\n"
        "Rules:\n"
        "- 'avg rent', 'how much', 'summary' -> get_property_summary\n"
        "- 'unit mix', 'breakdown by type'   -> get_unit_mix\n"
        "- 'occupancy', 'vacant'             -> get_occupancy (pass month if mentioned, format YYYY-MM)\n"
        "- 'trend', 'over the year', 'monthly'-> get_rent_trend (months=12 default)\n"
        "- 'expiring', 'renewals'            -> get_expiring_leases (within_days default 90)\n"
        "- 'balance', 'delinquent', 'owed'   -> get_top_balances (n default 10)\n"
        "Never include property_code in args — it is injected automatically."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]


def sql_node(state: ChatState) -> dict[str, Any]:
    """Use the LLM to pick a SQL tool, then run it (scoped to property)."""
    provider = get_provider(state["llm_provider"])
    model = validate_model(state["llm_provider"], state["model"])

    messages = _sql_tool_picker_prompt(state["user_message"], state["property_code"])
    raw = provider.generate(
        messages=messages,
        model=model,
        temperature=0.0,
        response_format="json_object",
        max_tokens=200,
    )
    # Be lenient: strip markdown fences if any provider leaked them.
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    tool_name = "get_property_summary"
    args: dict[str, Any] = {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("tool") in SQL_TOOLS:
            tool_name = parsed["tool"]
            args = parsed.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            # Strip any sneaky property_code in args — we inject it.
            args.pop("property_code", None)
    except json.JSONDecodeError:
        pass  # fall back to summary

    try:
        result = run_sql_tool(tool_name, state["property_code"], **args)
    except TypeError:
        # bad kwarg from the LLM -> retry with no args
        result = run_sql_tool(tool_name, state["property_code"])
        args = {}

    return {"sql_tool": tool_name, "sql_args": args, "sql_result": result}


# ---------------------------------------------------------------------------
# 4. rag_node — vector search filtered by property_code
# ---------------------------------------------------------------------------

def rag_node(state: ChatState) -> dict[str, Any]:
    r = search_property(state["property_code"], state["user_message"], k=4)
    return {"rag_chunks": r["chunks"], "rag_sources": r["sources"]}


# ---------------------------------------------------------------------------
# 5. hybrid_node — both
# ---------------------------------------------------------------------------

def hybrid_node(state: ChatState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(sql_node(state))
    out.update(rag_node(state))
    return out


# ---------------------------------------------------------------------------
# 6. compose_response — LLM writes the narrative; we attach components
# ---------------------------------------------------------------------------

_COMPOSE_SYSTEM_FOOTER = """\

Output requirements:
- Reply in Markdown. Use short headings, bullets, bold for numbers.
- DO NOT repeat the property name in every sentence.
- If the SQL tool returned no data, say so plainly.
- Cite RAG sources inline as [Source N] when used.
"""


def _compose_user_block(state: ChatState) -> str:
    parts = [f"User question: {state['user_message']}"]

    sql_result = state.get("sql_result")
    sql_tool = state.get("sql_tool")
    if sql_result is not None:
        parts.append(
            f"\nSQL tool called: `{sql_tool}` with args={state.get('sql_args', {})}\n"
            "SQL result (JSON):\n```json\n"
            + json.dumps(sql_result, indent=2, default=str)
            + "\n```"
        )

    rag_chunks = state.get("rag_chunks") or []
    if rag_chunks:
        parts.append("\nRetrieved property-website excerpts:\n" + build_context_block(rag_chunks))

    if not sql_result and not rag_chunks:
        parts.append("\n(No tool results available — answer with what you can or say so.)")

    return "\n".join(parts)


def compose_response(state: ChatState) -> dict[str, Any]:
    provider = get_provider(state["llm_provider"])
    model = validate_model(state["llm_provider"], state["model"])

    sys_prompt = system_prompt(state["property_code"], state["property_name"])
    messages = [
        {"role": "system", "content": sys_prompt + _COMPOSE_SYSTEM_FOOTER},
        {"role": "user", "content": _compose_user_block(state)},
    ]
    answer = provider.generate(
        messages=messages,
        model=model,
        temperature=0.3,
        max_tokens=900,
    )

    components = build_components(state.get("sql_tool"), state.get("sql_result"))
    sources = state.get("rag_sources") or []

    return {
        "answer_markdown": answer.strip(),
        "components": components,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def route_selector(state: ChatState) -> str:
    """Pick which branch to execute based on classify_query's decision."""
    return state.get("route", "sql")
