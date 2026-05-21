"""Graph nodes (v2 rewrite) — canonical LangGraph tool-calling agent.

Topology:

    START
      ▼
    extract_scope ──(conflict|missing)──▶ clarify (interrupt)
         │                                    │
         │                                    └── resumed ───┐
         ▼ (single|compare)                                   │
       seed_messages ◀────────────────────────────────────────┘
         ▼
       agent ◀────────┐
         │             │
       (tool_calls?)   │
         ├─yes─▶ tools ┘   (executes EVERY tool_call in parallel)
         └─no ─▶ compose ─▶ END

Design notes (post-rewrite):

  1. State uses `MessagesState`-style `messages` with `add_messages` reducer.
     The LLM conversation IS the state — agent/tools just append.
  2. The `tools` node runs ALL tool_calls in the last AIMessage. This is the
     standard parallel tool-calling pattern and is mandatory for letting the
     LLM emit, e.g., `compare_units` AND `render_chart` in a single turn.
  3. No `critic` node. Native tool-calling LLMs already signal "I'm done"
     by returning an AIMessage with no tool_calls. We honor that and add a
     `max_turns` hard cap as the only retry/give-up control.
  4. `tool_history` (separate from `messages`) keeps full untruncated results
     for the UI Tool Trace, while `ToolMessage.content` is JSON-truncated
     to fit the prompt budget.
  5. The tools list is built ONCE per request — closures capture scope.
"""
from __future__ import annotations

import json
import re
import time
import uuid as _uuid
from typing import Any

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from ..config import get_settings
from ..llm_registry import ProviderUnavailable
from ..guardrails.scope import (
    ScopeDecision,
    UnknownPropertyError,
    resolve_scope,
    system_prompt,
    validate_property_code,
)
from ..tools.sql_tools import TOOLS as SQL_TOOLS
from ..tools.rag_tools import search_property
from .components import build_components
from .state import ChatState, ToolStep


_settings = get_settings()
MAX_TURNS_DEFAULT = 8
TOOL_CONTENT_PREVIEW_CHARS = 4000   # cap on ToolMessage.content serialisation


# ---------------------------------------------------------------------------
# Chart rendering — type aliases + shape normalisation
# ---------------------------------------------------------------------------
#
# The LLM frequently emits short-form chart types ("pie" instead of
# "pie_chart") or the wrong field names for the chart family it picked
# ({x, y} for a pie that wants {labels, values}). Both used to silently
# drop the chart from the response. Normalisation gives the LLM slack
# without compromising the UI contract.

ALLOWED_CHART_TYPES = frozenset({
    "kpi", "table", "bar_chart", "line_chart",
    "comparison_chart", "pie_chart", "donut_chart",
})

CHART_TYPE_ALIASES = {
    "pie":               "pie_chart",
    "doughnut":          "donut_chart",
    "donut":             "donut_chart",
    "bar":               "bar_chart",
    "barchart":          "bar_chart",
    "line":              "line_chart",
    "linechart":         "line_chart",
    "compare":           "comparison_chart",
    "comparison":        "comparison_chart",
    "compare_chart":     "comparison_chart",
}


def _normalise_chart_data(chart_type: str, data: dict) -> dict:
    """Translate common LLM-mistake shapes into the canonical shape for the type.

    Examples handled:
      - pie/donut with {x, y}             -> {labels: x, values: y}
      - bar/line with {labels, values}    -> {x: labels, y: values}
      - comparison_chart with {labels, values}  -> wraps as a single-row series
    """
    if not isinstance(data, dict):
        return data

    if chart_type in ("pie_chart", "donut_chart"):
        if "labels" not in data and "x" in data:
            data = {**data, "labels": data["x"]}
        if "values" not in data and "y" in data:
            data = {**data, "values": data["y"]}
        if "values" not in data and "data" in data and isinstance(data["data"], list):
            # {labels: [...], data: [{name,value}, ...]} → flat values list
            data = {**data, "values": [d.get("value") for d in data["data"]]}

    elif chart_type in ("bar_chart", "line_chart"):
        if "x" not in data and "labels" in data:
            data = {**data, "x": data["labels"]}
        if "y" not in data and "values" in data:
            data = {**data, "y": data["values"]}

    elif chart_type == "comparison_chart":
        if "categories" not in data and "labels" in data:
            data = {**data, "categories": data["labels"]}
        if "rows" not in data and "values" in data:
            cats = data.get("categories") or []
            vals = data.get("values") or []
            data = {
                **data,
                "rows": [{"dimension": "value",
                          **{c: v for c, v in zip(cats, vals)}}],
            }

    return data


# Minimum-required keys per chart type (after normalisation).
_CHART_SHAPE = {
    "pie_chart":         ("labels", "values"),
    "donut_chart":       ("labels", "values"),
    "bar_chart":         ("x", "y"),
    "line_chart":        ("x", "y"),
    "comparison_chart":  ("categories", "rows"),
    "table":             ("columns", "rows"),
    "kpi":               ("value",),
}


def _resolve_chart_title(
    explicit: str | None,
    chart_type: str,
    data: dict,
) -> str:
    """Pick a sensible title when the LLM didn't supply one.

    Order of preference:
      1. The caller's explicit `title` if non-empty.
      2. A title-like field inside `data` (label/title/name/metric/subtitle).
         These are common places the LLM stuffs the label when it forgets
         the top-level `title` arg.
      3. A reasonable default derived from the chart type.

    The fallback path uses the most specific data field available so a
    series of KPI cards with distinct subtitles produces distinct titles
    and isn't collapsed by the (type, title) dedup in _collect_chart_attempts.
    """
    if explicit and isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if isinstance(data, dict):
        for key in ("title", "label", "name", "metric", "subtitle", "header"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # KPI: the `value` itself isn't a title, but if nothing else, prepend
        # "Metric" to give the dedup key uniqueness when subtitles also exist.
        if chart_type == "kpi" and data.get("value") is not None:
            return f"KPI · {data['value']}"
    return chart_type.replace("_", " ").title()


def _validate_chart_data(chart_type: str, data: dict) -> tuple[bool, str]:
    """Return (is_valid, error_or_'')."""
    if not isinstance(data, dict):
        return False, "data must be a dict"
    expected = _CHART_SHAPE.get(chart_type, ())
    missing = [k for k in expected if k not in data]
    if missing:
        return False, f"missing required keys for {chart_type}: {missing}; expected {list(expected)}"
    # For visual charts, require non-empty values.
    if chart_type in ("pie_chart", "donut_chart"):
        if not data.get("labels") or not data.get("values"):
            return False, "pie/donut needs non-empty labels and values"
    return True, ""


# ---------------------------------------------------------------------------
# Tool factory — closures capture scope so the LLM never supplies it.
# ---------------------------------------------------------------------------

def _build_tools(scope: ScopeDecision):
    """Build the LangChain @tool list for a given resolved scope.

    Returns (tool_list, dispatch_map). `dispatch_map` is `{name: BaseTool}` —
    used by the `tools` node to invoke a tool_call by name with full error
    capture.
    """
    primary_code = scope.code or (scope.codes[0] if scope.codes else None)
    scope_codes = scope.codes if scope.kind == "compare" else (
        [scope.code] if scope.code else []
    )

    @tool
    def get_property_summary() -> dict:
        """High-level KPIs for the latest monthly snapshot: unit count, occupancy %, avg rent, total rent roll."""
        return SQL_TOOLS["get_property_summary"]["fn"](primary_code)

    @tool
    def get_unit_mix() -> dict:
        """Breakdown by unit_type (count, avg market rent, avg sqft) for the active property."""
        return SQL_TOOLS["get_unit_mix"]["fn"](primary_code)

    @tool
    def get_occupancy(month: str | None = None) -> dict:
        """Occupancy % for a single month (YYYY-MM). Defaults to latest snapshot."""
        return SQL_TOOLS["get_occupancy"]["fn"](primary_code, month=month)

    @tool
    def get_rent_trend(months: int = 12) -> dict:
        """Monthly avg-rent and occupancy time series. `months` clamps the window (default 12)."""
        return SQL_TOOLS["get_rent_trend"]["fn"](primary_code, months=months)

    @tool
    def get_expiring_leases(within_days: int = 90, reference_date: str | None = None) -> dict:
        """Leases expiring within N days of `reference_date` (default: today). Returns rows ordered by lease_end."""
        return SQL_TOOLS["get_expiring_leases"]["fn"](
            primary_code, within_days=within_days, reference_date=reference_date
        )

    @tool
    def get_top_balances(n: int = 10) -> dict:
        """Top N leases by outstanding balance (most owed first). Default n=10."""
        return SQL_TOOLS["get_top_balances"]["fn"](primary_code, n=n)

    @tool
    def get_unit_charges(unit_number: str, snapshot_month: str | None = None) -> dict:
        """Every charge line item for ONE unit, in source order. Preserves multiplicity — e.g. two PARKING lines appear separately. Use this for 'what fees does unit X pay?' / 'how much parking does A103 pay?' questions."""
        return SQL_TOOLS["get_unit_charges"]["fn"](
            primary_code, unit_number=unit_number, snapshot_month=snapshot_month
        )

    @tool
    def compare_units(unit_numbers: list[str], dimensions: list[str] | None = None) -> dict:
        """Side-by-side comparison of 2+ units WITHIN the active property. dimensions can include rent, sqft, market_rent, balance, bedrooms, bathrooms."""
        return SQL_TOOLS["compare_units"]["fn"](
            primary_code, unit_numbers=unit_numbers, dimensions=dimensions
        )

    @tool
    def compare_properties(dimension: str = "avg_rent", month: str | None = None) -> dict:
        """Aggregate one metric across the active comparison property codes. Use when the user asks to compare 2+ PROPERTIES (e.g. '115r vs 126r'). dimension: avg_rent | occupancy_pct | total_units | occupied_units | rent_roll_total."""
        if scope.kind != "compare":
            return {"error": "compare_properties requires 2+ property codes in scope. Currently scoped to one property."}
        return SQL_TOOLS["compare_properties"]["fn"](
            property_codes=scope.codes, dimension=dimension, month=month
        )

    @tool
    def list_units(
        unit_type: str | None = None,
        bedrooms: float | None = None,
        min_rent: float | None = None,
        max_rent: float | None = None,
        occupied: bool | None = None,
        lease_ends_before: str | None = None,
        lease_ends_after: str | None = None,
        limit: int = 50,
    ) -> dict:
        """Filtered list of units. Combine any of: unit_type, bedrooms, min_rent, max_rent, occupied, lease_ends_before/after (YYYY-MM-DD)."""
        return SQL_TOOLS["list_units"]["fn"](
            primary_code,
            unit_type=unit_type, bedrooms=bedrooms,
            min_rent=min_rent, max_rent=max_rent, occupied=occupied,
            lease_ends_before=lease_ends_before, lease_ends_after=lease_ends_after,
            limit=limit,
        )

    @tool
    def execute_scoped_sql(sql: str) -> dict:
        """BACKSTOP — run a custom read-only SELECT against the rent-roll DB. Use when no curated tool fits a complex multi-condition question. Tables: properties, units, leases, rent_snapshots, rent_charge_lines. Property scope is auto-injected; do NOT add property_code filters yourself. Validated by sqlglot and executed as a read-only DB user."""
        return SQL_TOOLS["execute_scoped_sql"]["fn"](scope_codes, sql)

    @tool
    def search_property_pages(query: str, k: int = 4) -> dict:
        """Semantic search over the active property's MARKETING WEBSITE chunks (amenities, floor plans, pet policy, etc). Use for qualitative questions ('what amenities does it offer?', 'pet policy', 'community features'). Returns text chunks + source URLs."""
        if not primary_code:
            return {"error": "search_property_pages requires a single property in scope"}
        return search_property(primary_code, query, k=k)

    @tool
    def render_chart(
        chart_type: str,
        data: dict,
        title: str | None = None,
    ) -> dict:
        """Render a UI chart / table / KPI card from data you ALREADY have.

        IMPORTANT: this is HOW you draw a visual. Calling it makes a chart
        appear in the user's UI. Never say "I cannot draw" — call this tool
        instead. You may emit it IN PARALLEL with a data tool in the same
        turn (e.g. compare_units + render_chart together). For dashboards,
        call render_chart MULTIPLE times with distinct titles to surface
        several KPI cards / charts side by side.

        chart_type (canonical names, but short aliases like "pie" / "bar" /
        "line" / "donut" / "compare" also work):
          pie_chart        data: labels=[..], values=[..]
          donut_chart      data: labels=[..], values=[..]
          bar_chart        data: x=[..], y=[..]
          line_chart       data: x=[..], y=[..], y_label?, secondary?: {label, y}
          comparison_chart data: categories=[..], rows=[{dimension, <cat>:val, ...}]
          table            data: columns=[..], rows=[[..], ..]
          kpi              data: value="..", subtitle?: ".."
                            (Provide ONE metric per kpi card. For multiple
                             metrics, call render_chart once per card with
                             a distinct title.)

        `title` is the heading shown above the visual (e.g. "Avg Rent" for
        a KPI card or "Rent Comparison" for a chart). If you omit it, a
        title is inferred from `data` (data.label, data.subtitle, data.name)
        — but providing an explicit, descriptive title is strongly preferred.
        """
        # 1. Accept short-form aliases ("pie" → "pie_chart" etc.)
        raw = (chart_type or "").strip().lower()
        normalised_type = CHART_TYPE_ALIASES.get(raw, raw)
        if normalised_type not in ALLOWED_CHART_TYPES:
            return {
                "error": (
                    f"chart_type must be one of {sorted(ALLOWED_CHART_TYPES)} "
                    f"(or a short alias like 'pie', 'bar', 'line'); got {chart_type!r}"
                ),
                "attempted_type": chart_type,
            }
        # 2. Normalise the data shape for common LLM-mistake patterns.
        norm_data = _normalise_chart_data(normalised_type, data or {})
        # 3. Infer a title if the LLM omitted it. Different defaults per type
        #    so deduplication doesn't collapse legitimately-distinct KPIs.
        resolved_title = _resolve_chart_title(title, normalised_type, norm_data)
        # 4. Validate the data shape AFTER normalisation.
        ok, err = _validate_chart_data(normalised_type, norm_data)
        if not ok:
            return {
                "error": f"data shape invalid for {normalised_type}: {err}",
                "attempted_type": normalised_type,
                "attempted_title": resolved_title,
                "received_keys": list((data or {}).keys()) if isinstance(data, dict) else [],
            }
        return {
            "chart_spec": {"type": normalised_type, "title": resolved_title, "data": norm_data},
            "ok": True,
        }

    tool_list = [
        get_property_summary, get_unit_mix, get_occupancy, get_rent_trend,
        get_expiring_leases, get_top_balances, get_unit_charges,
        compare_units, compare_properties, list_units, execute_scoped_sql,
        search_property_pages, render_chart,
    ]
    return tool_list, {t.name: t for t in tool_list}


# ---------------------------------------------------------------------------
# Per-invocation cache: build tools once per (conversation_id, scope key).
# Without this, every node call rebuilds the closure-heavy tool list.
# ---------------------------------------------------------------------------

_TOOLS_CACHE: dict[tuple, tuple[list, dict]] = {}


def _tools_for(state: ChatState):
    scope = ScopeDecision.from_dict(state["scope"])
    key = (state.get("conversation_id"), scope.kind, scope.code, tuple(scope.codes or []))
    cached = _TOOLS_CACHE.get(key)
    if cached is not None:
        return cached
    built = _build_tools(scope)
    # Bounded cache (avoid leaks across many conversations)
    if len(_TOOLS_CACHE) > 64:
        _TOOLS_CACHE.clear()
    _TOOLS_CACHE[key] = built
    return built


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _scope_summary(scope: dict) -> str:
    kind = scope.get("kind")
    if kind == "single":
        return f"single property {scope.get('code')!r}"
    if kind == "compare":
        return f"comparison across properties {scope.get('codes')}"
    return f"unresolved ({kind})"


def _build_system_prompt(state: ChatState) -> str:
    """One prompt, built once per request from scope + property_name."""
    scope = state.get("scope") or {}
    scope_summary = _scope_summary(scope)

    base = (
        "You are a property-management AI analyst. You answer using the "
        "bound tools — do not invent numbers.\n\n"
        f"Active scope: {scope_summary}.\n"
    )
    if scope.get("kind") == "single" and state.get("property_name"):
        base += f"Active property name: {state['property_name']}.\n"

    base += (
        "\nTool selection guide (use the MOST specific):\n"
        "  - Specific unit fees / charges ('parking for A103')      → get_unit_charges\n"
        "  - Compare 2+ units in same property                       → compare_units\n"
        "  - Compare 2+ properties on a metric                       → compare_properties\n"
        "  - 'average rent / occupancy / summary'                    → get_property_summary\n"
        "  - Unit-type breakdown                                     → get_unit_mix\n"
        "  - Monthly trend / over the year                           → get_rent_trend\n"
        "  - Expiring leases                                         → get_expiring_leases\n"
        "  - High balances / delinquent                              → get_top_balances\n"
        "  - Filtered unit list (multi-condition)                    → list_units\n"
        "  - NOVEL queries no curated tool covers                    → execute_scoped_sql\n"
        "  - Amenities / floor plans / pet policy / marketing copy   → search_property_pages\n"
        "\nKPI cards — read carefully:\n"
        "  - When the user asks for 'KPI(s)', 'key metrics', or a 'dashboard', "
        "emit ONE render_chart(chart_type='kpi') call PER METRIC with a "
        "distinct, descriptive title. ALWAYS pass `title` explicitly.\n"
        "  - Example: title='A103 · Monthly Rent', "
        "data={value:'$2,480', subtitle:'755 sqft, lease ends 2026-06-06'}.\n"
        "  - For unit/property comparisons, emit one KPI card per "
        "(entity, metric) pair — six cards for A103 vs A107 across rent / "
        "sqft / market rent.\n"
        "  - Never bury KPI values inside a markdown bullet list. The user "
        "asked for cards.\n"
        "\nCharts — read carefully:\n"
        "  - The user's UI renders any chart you emit via render_chart. They will SEE the chart.\n"
        "  - If the user mentions a chart type (pie, bar, line, donut, chart, plot, graph), "
        "you MUST call render_chart with that type before finishing.\n"
        "  - ALWAYS pass `title` as a top-level argument to render_chart.\n"
        "  - Prefer emitting the DATA tool and render_chart IN PARALLEL in the same turn — "
        "this is supported and faster.\n"
        "  - Never tell the user 'I cannot draw' — render_chart IS how you draw.\n"
        "  - Never embed inline images (no ![](...) markdown, no base64) in your final reply. "
        "The chart is already on screen from render_chart.\n"
        "  - Call render_chart AT MOST ONCE per distinct chart_type+title. Don't repeat the "
        "same chart with different data — get the data right first, THEN render.\n"
        "  - In your final natural-language reply, refer to the chart by title only "
        "(e.g. \"the pie chart above shows…\"). Just narrate the numbers.\n"
        "\nMulti-step reasoning:\n"
        "  - Decompose complex requests BEFORE charting. Example: for "
        "'compare A103 and A104', if `compare_units` returns "
        "`missing: ['A104']`, DO NOT refuse the request — chart A103 and "
        "say in your reply that A104 was not found.\n"
        "  - Treat each user message as a fresh question that may reference "
        "prior conversation context. Read the full message history.\n"
        "  - If a unit, month, or property the user named doesn't exist in "
        "the data, still answer with what IS available and call out what's "
        "missing — don't silently drop the question.\n"
        "\nFinish: stop calling tools and reply with a brief natural-language summary "
        "once you have enough data AND have rendered any charts the user asked for."
    )
    return base


_COMPOSER_SYSTEM = (
    "You are a property-management analyst writing the FINAL ANSWER for the user.\n\n"
    "Hard rules:\n"
    "1. Stay strictly within the active scope.\n"
    "2. Only use the data shown in the tool history below. Do NOT invent numbers.\n"
    "3. Reply in Markdown. Use short headings, bullets, **bold** for numbers.\n"
    "4. When a tool returned MULTIPLE rows with the same charge code (e.g. two PARKING "
    "lines), list them individually with their amounts. Do not collapse them into a sum.\n"
    "5. If the agent gave up (max turns reached or all attempts failed), write a brief "
    "honest transcript: which approaches were tried and why each failed.\n"
    "6. The UI is rendering any charts already. Don't say 'I cannot draw' — refer to "
    "the chart by title if useful, otherwise just narrate the numbers."
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_llm(state: ChatState):
    """Return a ChatModel for the user's selected provider+model.

    Honors `state["llm_provider"]` (set from the frontend dropdown). All
    three providers use their native LangChain chat classes, which all
    implement `.bind_tools(...)` with the same surface — so the
    agent/tools loop works identically regardless of provider.

    Raises ProviderUnavailable when the API key for the selected provider
    isn't configured. main.py catches that and returns HTTP 400.
    """
    provider = (state.get("llm_provider") or "openai").lower()
    model = state.get("model") or "gpt-4o-mini"
    temperature = 0.2

    if provider == "openai":
        if not _settings.openai_api_key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        return ChatOpenAI(
            model=model, temperature=temperature,
            api_key=_settings.openai_api_key,
        )

    if provider == "anthropic":
        if not _settings.anthropic_api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model, temperature=temperature,
            api_key=_settings.anthropic_api_key,
        )

    if provider == "gemini":
        if not _settings.google_api_key:
            raise ProviderUnavailable("GOOGLE_API_KEY not set")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature,
            google_api_key=_settings.google_api_key,
        )

    raise ProviderUnavailable(f"Unknown provider: {provider!r}")


def _truncate(s: str, n: int = TOOL_CONTENT_PREVIEW_CHARS) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n… (truncated {len(s) - n} chars)"


def _serialise_for_tool_message(result: Any) -> str:
    try:
        s = json.dumps(result, default=str, ensure_ascii=False)
    except Exception:
        s = str(result)
    return _truncate(s)


# ---------------------------------------------------------------------------
# 1. extract_scope
# ---------------------------------------------------------------------------

def extract_scope(state: ChatState) -> dict[str, Any]:
    decision = resolve_scope(
        dropdown_code=state.get("dropdown_property_code"),
        message=state.get("user_message", ""),
    )
    out: dict[str, Any] = {"scope": decision.to_dict()}
    if decision.kind == "single" and decision.code:
        try:
            _, name = validate_property_code(decision.code)
            out["property_name"] = name
        except UnknownPropertyError:
            out["scope"] = {**decision.to_dict(), "kind": "missing"}
    return out


def scope_router(state: ChatState) -> str:
    kind = (state.get("scope") or {}).get("kind", "missing")
    if kind in {"conflict", "missing"}:
        return "clarify"
    return "enter_turn"


# ---------------------------------------------------------------------------
# 2. clarify — LangGraph interrupt; resumes here with the user's reply
# ---------------------------------------------------------------------------

def clarify(state: ChatState) -> dict[str, Any]:
    scope = state.get("scope") or {}
    kind = scope.get("kind")

    if kind == "conflict":
        question = (
            f"Your message mentioned property **{scope.get('query_code')}** "
            f"but the dropdown is set to **{scope.get('dropdown_code')}**. "
            f"Which property should I use?"
        )
        options = [scope.get("query_code"), scope.get("dropdown_code")]
    elif kind == "missing":
        question = (
            "Which property are you asking about? I couldn't find a property "
            "code in your message and no property is selected."
        )
        options = scope.get("available") or []
    else:
        question = "Please confirm the property to use."
        options = []

    user_choice = interrupt({
        "question": question,
        "options": [o for o in options if o],
        "scope_kind": kind,
    })

    if isinstance(user_choice, str):
        chosen = [c.strip().lower() for c in user_choice.split(",") if c.strip()]
    elif isinstance(user_choice, list):
        chosen = [c.strip().lower() for c in user_choice if isinstance(c, str)]
    else:
        chosen = []

    if not chosen:
        from ..guardrails.scope import _all_property_codes
        return {"scope": ScopeDecision(kind="missing", available=_all_property_codes()).to_dict()}

    valid: list[str] = []
    for c in chosen:
        try:
            validate_property_code(c)
            valid.append(c)
        except UnknownPropertyError:
            continue

    if len(valid) >= 2:
        return {"scope": ScopeDecision(kind="compare", codes=valid, source="resumed").to_dict()}
    if len(valid) == 1:
        _, name = validate_property_code(valid[0])
        return {
            "scope": ScopeDecision(kind="single", code=valid[0], source="resumed").to_dict(),
            "property_name": name,
        }

    from ..guardrails.scope import _all_property_codes
    return {"scope": ScopeDecision(kind="missing", available=_all_property_codes()).to_dict()}


# ---------------------------------------------------------------------------
# 3. enter_turn — open a new conversational turn
# ---------------------------------------------------------------------------
#
# Runs once at the start of every /chat invocation (after scope resolution).
# This is the fix for the multi-turn persistence bug: it ALWAYS appends the
# new user message to `messages`. The SystemMessage is seeded only on the
# very first turn. Per-turn state (turn_count, tool_history, components,
# answer_markdown, sources, route, gave_up) is reset so the response shows
# only what happened in THIS turn.

# Marker the scope-refresh SystemMessage carries so we can detect drift
# across turns without re-parsing the full system prompt every time.
_SCOPE_MARKER = "[scope]"


def _last_scope_summary_in(messages: list) -> str | None:
    """Find the most recent scope summary embedded in any prior SystemMessage."""
    for m in reversed(messages):
        if isinstance(m, SystemMessage) and _SCOPE_MARKER in (m.content or ""):
            # SystemMessage content has the marker followed by the summary text.
            try:
                return (m.content.split(_SCOPE_MARKER, 1)[1]).strip().splitlines()[0]
            except (IndexError, AttributeError):
                continue
    return None


def enter_turn(state: ChatState) -> dict[str, Any]:
    """Open a new conversational turn.

    - First turn: seed the full SystemMessage + the user's HumanMessage.
    - Later turns: append only the new HumanMessage. If the active scope
      drifted from the prior turn (different property), append a small
      'scope refresh' SystemMessage so the LLM knows.
    - Resets per-turn state regardless.
    """
    existing = state.get("messages") or []
    current_summary = _scope_summary(state.get("scope") or {})
    last_summary = _last_scope_summary_in(existing)

    new_msgs: list = []
    if not existing or last_summary is None:
        # First turn — seed the full system prompt. The marker line at the
        # very top makes scope-drift detection cheap on later turns.
        sys_text = (
            f"{_SCOPE_MARKER} {current_summary}\n\n" + _build_system_prompt(state)
        )
        new_msgs.append(SystemMessage(content=sys_text))
    elif last_summary != current_summary:
        # Scope drifted between turns — append a refresh note rather than a
        # full re-prompt. Positioned at the end so it's the most-recent
        # instruction the LLM sees.
        new_msgs.append(SystemMessage(
            content=f"{_SCOPE_MARKER} {current_summary}\n\n"
                    f"Scope updated for this turn: {current_summary}. "
                    "Use ONLY this scope when answering the next user message."
        ))
    new_msgs.append(HumanMessage(content=state["user_message"]))

    return {
        "messages": new_msgs,                       # add_messages reducer appends
        "turn_count": 0,                            # per-turn counter resets
        "max_turns": state.get("max_turns") or MAX_TURNS_DEFAULT,
        # Per-turn outputs reset; turn-local trace (custom reducer treats [] as clear)
        "tool_history": [],
        "answer_markdown": "",
        "components": [],
        "sources": [],
        "route": "agent",
        "gave_up": False,
    }


# ---------------------------------------------------------------------------
# 4. agent — the LLM turn
# ---------------------------------------------------------------------------

def agent(state: ChatState) -> dict[str, Any]:
    tool_list, _ = _tools_for(state)
    llm = _make_llm(state).bind_tools(tool_list)
    ai_msg = llm.invoke(state["messages"])
    # Ensure every tool_call has a stable id (some providers omit it).
    if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            if not tc.get("id"):
                tc["id"] = f"call_{_uuid.uuid4().hex[:12]}"
    return {
        "messages": [ai_msg],
        "turn_count": state.get("turn_count", 0) + 1,
    }


def agent_should_continue(state: ChatState) -> str:
    """Decide: more tools, or compose the answer?"""
    if state.get("turn_count", 0) >= state.get("max_turns", MAX_TURNS_DEFAULT):
        return "compose"  # hard stop
    msgs: list[BaseMessage] = state.get("messages") or []
    if not msgs:
        return "compose"
    last = msgs[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "compose"


# ---------------------------------------------------------------------------
# 5. tools — run EVERY tool_call in the last AIMessage (parallel)
# ---------------------------------------------------------------------------

def tools(state: ChatState) -> dict[str, Any]:
    _, dispatch = _tools_for(state)
    last_ai: AIMessage | None = None
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    if last_ai is None or not last_ai.tool_calls:
        return {}

    new_messages: list[BaseMessage] = []
    new_history: list[ToolStep] = []

    for tc in last_ai.tool_calls:
        name = tc.get("name")
        args = tc.get("args") or {}
        call_id = tc.get("id") or f"call_{_uuid.uuid4().hex[:12]}"

        started = time.monotonic()
        ok = False
        result: Any = None
        error: str | None = None
        if name not in dispatch:
            error = f"Unknown tool: {name}"
            result = {"error": error}
        else:
            try:
                result = dispatch[name].invoke(args)
                if isinstance(result, dict) and result.get("error"):
                    error = str(result["error"])
                else:
                    ok = True
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                result = {"error": error}
        duration_ms = int((time.monotonic() - started) * 1000)

        # The LLM sees the truncated string; UI sees the full dict.
        content_str = _serialise_for_tool_message(result)
        new_messages.append(ToolMessage(
            content=content_str,
            tool_call_id=call_id,
            name=name,
        ))
        new_history.append({
            "tool": name,
            "args": args,
            "ok": ok,
            "result": result,
            "error": error,
            "duration_ms": duration_ms,
            "tool_call_id": call_id,
        })

    return {"messages": new_messages, "tool_history": new_history}


# ---------------------------------------------------------------------------
# 6. compose — final markdown + components
# ---------------------------------------------------------------------------

def _collect_chart_attempts(history: list[ToolStep]) -> tuple[list[dict], list[dict]]:
    """Inspect history for render_chart calls.

    Returns (rendered_specs, failed_attempts):
      - rendered_specs: deduped by (type, title), latest wins. Order of
        first occurrence preserved for stable UI layout.
      - failed_attempts: list of {attempted_type, error, args} for chart
        calls that returned an error. Surfaced in compose so the user
        learns "your chart didn't render and here's why" instead of being
        silently dropped.
    """
    by_key: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    failures: list[dict] = []

    for step in history:
        if step.get("tool") != "render_chart":
            continue
        result = step.get("result")
        if step.get("ok") and isinstance(result, dict):
            spec = result.get("chart_spec")
            if isinstance(spec, dict) and spec.get("type") and spec.get("title"):
                key = (spec["type"], spec["title"])
                if key not in by_key:
                    order.append(key)
                by_key[key] = {
                    "type": spec["type"],
                    "title": spec["title"],
                    "data": spec.get("data") or {},
                }
                continue
        # Anything else under render_chart is a failure.
        failures.append({
            "attempted_type": (result or {}).get("attempted_type") if isinstance(result, dict) else None,
            "args": step.get("args"),
            "error": step.get("error") or (result or {}).get("error") if isinstance(result, dict) else None,
        })

    return [by_key[k] for k in order], failures


# Strip hallucinated inline images: `![alt](data:image/...;base64,...)` etc.
# The agent's prompt forbids these, but defense-in-depth keeps the UI clean
# even if it slips. We DO allow legit external image URLs (rare but possible
# for property marketing images surfaced by RAG).
_INLINE_IMG_RE = re.compile(
    r"!\[[^\]]*\]\((?:data:|<svg)[^)]*\)",
    re.IGNORECASE | re.DOTALL,
)


def _scrub_inline_images(text: str) -> str:
    if not text:
        return text
    return _INLINE_IMG_RE.sub("", text).strip()


def _route_label(history: list[ToolStep]) -> str:
    sql_used = any(s.get("ok") and s.get("tool") in SQL_TOOLS for s in history)
    rag_used = any(s.get("ok") and s.get("tool") == "search_property_pages" for s in history)
    if sql_used and rag_used:
        return "hybrid"
    if rag_used:
        return "rag"
    if sql_used:
        return "sql"
    return "agent"


def compose(state: ChatState) -> dict[str, Any]:
    history: list[ToolStep] = state.get("tool_history") or []
    turn_count = state.get("turn_count", 0)
    max_turns = state.get("max_turns", MAX_TURNS_DEFAULT)
    gave_up = turn_count >= max_turns and not any(s.get("ok") for s in history)

    # The final AIMessage (no tool_calls) is the LLM's natural summary.
    final_ai_text = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage) and not m.tool_calls:
            final_ai_text = (m.content or "").strip()
            break

    # If the LLM gave a clean final answer AND it isn't empty, we can use it
    # directly. Otherwise (e.g. it stopped on a tool_calls AIMessage at the
    # turn cap), kick off a separate compose pass to get a real narrative.
    if final_ai_text and not gave_up:
        answer = final_ai_text
    else:
        llm = _make_llm(state)
        transcript_parts = [f"User question: {state.get('user_message', '')}"]
        for i, step in enumerate(history, 1):
            tag = "OK" if step.get("ok") else "ERROR"
            payload = step.get("result") if step.get("ok") else step.get("error")
            transcript_parts.append(
                f"\n[Step {i}] {step.get('tool')}({step.get('args')}) -> {tag}\n"
                f"{_truncate(json.dumps(payload, default=str), 3000)}"
            )
        if gave_up:
            transcript_parts.append(
                "\nNOTE: I hit the max turn cap without finding a working approach. "
                "Write an honest 'I tried these approaches' transcript."
            )
        answer = (llm.invoke([
            SystemMessage(content=_COMPOSER_SYSTEM),
            HumanMessage(content="\n".join(transcript_parts)),
        ]).content or "").strip()

    # Components precedence: explicit render_chart specs > deterministic
    # auto-emit for the last successful SQL tool. If an agent's render_chart
    # call failed (bad chart_type / shape), `chart_failures` carries the
    # detail so compose can surface a transparent note instead of silently
    # showing only the auto-emitted default.
    components, chart_failures = _collect_chart_attempts(history)
    if not components:
        last_ok_sql = next(
            (s for s in reversed(history)
             if s.get("ok") and s.get("tool") in SQL_TOOLS),
            None,
        )
        if last_ok_sql:
            components = build_components(
                last_ok_sql["tool"], last_ok_sql.get("result")
            ) or []

    # If render_chart was attempted but failed AND no successful chart of a
    # similar type was produced, surface the failure transparently. Better
    # to tell the user "your chart didn't render because X" than to silently
    # show an auto-emitted default or nothing at all.
    if chart_failures and not _has_chart_component(components):
        failure_note = _format_chart_failure_note(chart_failures)
        if failure_note:
            answer = (answer + "\n\n" + failure_note).strip()

    # Sources: from the last successful RAG step (if any).
    sources: list[dict] = []
    last_rag = next(
        (s for s in reversed(history)
         if s.get("ok") and s.get("tool") == "search_property_pages"),
        None,
    )
    if last_rag and isinstance(last_rag.get("result"), dict):
        sources = last_rag["result"].get("sources") or []

    return {
        "answer_markdown": _scrub_inline_images(answer),
        "components": components,
        "sources": sources,
        "route": _route_label(history),
        "gave_up": gave_up,
    }


def _has_chart_component(components: list[dict]) -> bool:
    """True if any of the emitted components is a chart (vs KPI/table)."""
    chart_types = {"pie_chart", "donut_chart", "bar_chart", "line_chart", "comparison_chart"}
    return any(c.get("type") in chart_types for c in components)


def _format_chart_failure_note(failures: list[dict]) -> str:
    """Concise Markdown note appended to the answer when a chart attempt failed."""
    if not failures:
        return ""
    lines = ["> _Note: a chart was requested but couldn't be drawn:_"]
    for f in failures[:3]:  # cap to avoid noise
        attempted = f.get("attempted_type") or "chart"
        err = (f.get("error") or "unknown error").strip()
        lines.append(f"> - `{attempted}` — {err}")
    return "\n".join(lines)
