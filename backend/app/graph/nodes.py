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
    validate_property_code,
)
from ..tools.sql_tools import TOOLS as SQL_TOOLS
from ..tools.rag_tools import search_property_active
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
    is_compare = scope.kind == "compare"

    def _pick_code(arg: str | None) -> str | dict:
        """Resolve which property the LLM is asking about.

        In compare mode the LLM MUST pass `property_code=` to single-property
        tools. If it's omitted we default to the primary (first) code but
        return a hint so the LLM can re-call for the other code. In single
        mode the arg is optional; mismatches raise.
        """
        if arg:
            c = arg.strip().lower()
            if c not in scope_codes:
                return {"error": f"property_code {arg!r} is not in the active scope ({scope_codes}). Use one of {scope_codes}."}
            return c
        return primary_code

    @tool
    def get_property_summary(property_code: str | None = None, snapshot_month: str | None = None) -> dict:
        """High-level KPIs (unit count, occupancy %, avg rent, total rent roll). Pass `snapshot_month` as YYYY-MM-DD for a specific monthly snapshot; omit for latest. In compare mode pass `property_code` to pick which property."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_property_summary"]["fn"](c, snapshot_month=snapshot_month)

    @tool
    def get_unit_mix(property_code: str | None = None) -> dict:
        """Breakdown by unit_type (count, avg market rent, avg sqft). In compare mode pass `property_code` to pick which property in scope."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_unit_mix"]["fn"](c)

    @tool
    def get_occupancy(month: str | None = None, property_code: str | None = None) -> dict:
        """Occupancy % for a single month (YYYY-MM). Defaults to latest snapshot. In compare mode pass `property_code` to pick which property."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_occupancy"]["fn"](c, month=month)

    @tool
    def get_rent_trend(months: int = 12, property_code: str | None = None) -> dict:
        """Monthly avg-rent and occupancy time series. `months` clamps the window (default 12). In compare mode pass `property_code` to pick which property."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_rent_trend"]["fn"](c, months=months)

    @tool
    def get_expiring_leases(within_days: int = 90, reference_date: str | None = None, property_code: str | None = None) -> dict:
        """Leases expiring within N days of `reference_date` (default: today). Returns rows ordered by lease_end. In compare mode pass `property_code` to pick which property."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_expiring_leases"]["fn"](
            c, within_days=within_days, reference_date=reference_date
        )

    @tool
    def get_top_balances(n: int = 10, snapshot_month: str | None = None, property_code: str | None = None) -> dict:
        """Top N leases by outstanding balance (most owed first). Default n=10. Pass `snapshot_month` as YYYY-MM-DD for a specific snapshot; omit for latest. In compare mode pass `property_code`."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_top_balances"]["fn"](c, n=n, snapshot_month=snapshot_month)

    @tool
    def get_lease_deposits(n: int = 50, snapshot_month: str | None = None, property_code: str | None = None) -> dict:
        """Resident Deposit + Other Deposit per lease + aggregate sums/averages. Pass `snapshot_month` as YYYY-MM-DD for a specific snapshot; omit for latest. In compare mode pass `property_code`."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_lease_deposits"]["fn"](c, n=n, snapshot_month=snapshot_month)

    @tool
    def get_move_outs(since: str | None = None, until: str | None = None, property_code: str | None = None) -> dict:
        """Leases with a Move Out date set, optionally filtered by date range.

DO NOT invent date filters for vague language. Pass `since`/`until` ONLY if the
user wrote an EXPLICIT calendar reference (a year, a month-name, a quarter, or
a literal date like '2026-01-01').

  - "who is moving out soon?"             → call with NO args (returns all)
  - "list 3 units moving out soon"        → call with NO args (returns all, agent picks 3)
  - "any tenants moving out?"             → call with NO args
  - "moving out this quarter"             → infer since/until of the CURRENT quarter
  - "moving out after January 2026"       → since='2026-01-01'
  - "moving out between Jan and Mar 2026" → since='2026-01-01', until='2026-03-31'

Both args are YYYY-MM-DD strings. In compare mode pass `property_code`."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_move_outs"]["fn"](c, since=since, until=until)

    @tool
    def get_unit_charges(unit_number: str, snapshot_month: str | None = None, property_code: str | None = None) -> dict:
        """Every charge line item for ONE unit, in source order. Preserves multiplicity — e.g. two PARKING lines appear separately. Use this for 'what fees does unit X pay?' / 'how much parking does A103 pay?' questions. In compare mode pass `property_code`."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["get_unit_charges"]["fn"](
            c, unit_number=unit_number, snapshot_month=snapshot_month
        )

    @tool
    def compare_units(unit_numbers: list[str], dimensions: list[str] | None = None, property_code: str | None = None) -> dict:
        """Side-by-side comparison of 2+ units WITHIN ONE property.

Dimensions: rent | monthly_rent | market_rent | sqft | balance | bedrooms | bathrooms.

CRITICAL — never invent unit numbers. Unit IDs vary wildly by property
(e.g. 175r uses LV01A-style codes, 115r uses A103-style, 134r uses S01-style).
If the user did NOT specify exact unit numbers — e.g. "compare any 2 units",
"pick 2 units and compare", "compare a couple of units" — you MUST first call
list_units(property_code=…, occupied=True, limit=10) to get REAL unit numbers
WITH active leases, then pick TWO from the returned rows that have non-null
`monthly_rent`. Don't pick units on notice / pending move-out (rent and lease
fields are often null for those — the comparison will be empty and useless).
THEN call compare_units with those real, data-rich values.

Calling compare_units with non-existent unit numbers returns an error and
you'll have to redo it. Calling it with notice-status units returns null
monthly_rent and you'll have to caveat your answer.

Numbers reflect the LATEST snapshot only (the `leases` / `units` tables).
ALWAYS state that in your reply (e.g. "as of the latest snapshot").

NULL handling: if a returned value is null (e.g. `monthly_rent` is often null
for units on notice / vacant / pending move-out) you MUST call that out
explicitly in your reply — say "monthly rent is not recorded for unit X
(notice status)" rather than leaving it as a silent gap. The chart drops
all-null dimensions automatically so users don't see empty bars.

In compare mode pass `property_code` to pick which property."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["compare_units"]["fn"](
            c, unit_numbers=unit_numbers, dimensions=dimensions
        )

    # --- DISABLED: cross-property comparison ---------------------------------
    # The property-vs-property comparison feature is out of scope for the
    # current assignment. The agent should only operate on a single property
    # at a time. Restore by uncommenting this @tool block AND its entry in
    # the returned tool list below (search for "compare_properties").
    #
    # @tool
    # def compare_properties(dimension: str = "avg_rent", month: str | None = None) -> dict:
    #     """Aggregate one metric across the active comparison property codes. Use when the user asks to compare 2+ PROPERTIES on a metric (e.g. '115r vs 126r avg rent'). dimension: avg_rent | occupancy_pct | total_units | occupied_units | rent_roll_total."""
    #     if scope.kind != "compare":
    #         return {"error": "compare_properties requires 2+ property codes in scope. Add more via the Property dropdown."}
    #     return SQL_TOOLS["compare_properties"]["fn"](
    #         property_codes=scope.codes, dimension=dimension, month=month
    #     )
    # -------------------------------------------------------------------------

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
        property_code: str | None = None,
    ) -> dict:
        """Filtered list of units. Combine any of: unit_type, bedrooms, min_rent, max_rent, occupied, lease_ends_before/after (YYYY-MM-DD). In compare mode pass `property_code` to pick which property in scope; otherwise defaults to the primary. To list units across BOTH properties in compare mode, call this tool TWICE (once per code)."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return SQL_TOOLS["list_units"]["fn"](
            c,
            unit_type=unit_type, bedrooms=bedrooms,
            min_rent=min_rent, max_rent=max_rent, occupied=occupied,
            lease_ends_before=lease_ends_before, lease_ends_after=lease_ends_after,
            limit=limit,
        )

    @tool
    def execute_scoped_sql(sql: str) -> dict:
        """BACKSTOP — run a custom read-only Postgres SELECT against the rent-roll DB. Use when no curated tool fits a complex multi-condition question. Property scope is auto-injected; do NOT add property_code filters yourself. Validated by sqlglot and executed as a read-only DB user.

SCHEMA (use these EXACT column names — do NOT invent columns):

  properties(
    property_code TEXT PK,
    property_name TEXT,
    property_type TEXT,  -- 'residential' | 'affordable' | 'commercial' | 'land' | etc.
    address TEXT
  )

  units(
    id INT PK,
    property_code TEXT FK,
    unit_number TEXT,
    unit_type TEXT,
    bedrooms FLOAT,
    bathrooms FLOAT,
    sqft FLOAT,
    market_rent FLOAT
  )

  leases(
    id INT PK,
    property_code TEXT FK,
    unit_number TEXT,
    tenant_id TEXT,
    lease_start DATE,
    lease_end DATE,
    monthly_rent FLOAT,    -- NOT `rent`; do not use `rent`
    balance FLOAT,         -- outstanding balance, latest snapshot only
    status TEXT,           -- 'current' | 'notice' | 'vacant' etc.
    resident_deposit FLOAT,
    other_deposit FLOAT,
    move_out_date DATE
  )

  rent_snapshots(
    id INT PK,
    property_code TEXT FK,
    snapshot_month DATE,   -- always the 1st of the month
    unit_number TEXT,
    monthly_rent FLOAT,
    occupied BOOLEAN,
    raw_row JSONB          -- per-unit raw row; access historical balance via (raw_row->>'balance')::numeric
  )                        -- NOTE: no `balance` column here; use raw_row JSONB

  rent_charge_lines(
    id INT PK,
    snapshot_id INT FK,
    property_code TEXT FK,
    snapshot_month DATE,
    unit_number TEXT,
    line_index INT,
    charge_code TEXT,      -- 'RENT' | 'PARKING' | 'PET' | 'AMENITY' | 'TRASH' | 'CONRENT' | …
    amount FLOAT
  )

Common gotchas:
  - For per-unit FEES (parking, pet, amenity, trash) use rent_charge_lines, NOT leases.
  - For historical balance use rent_snapshots.raw_row->>'balance', NOT a `balance` column.
  - For latest balance use leases.balance.
  - There is no `rent` column anywhere — it's `monthly_rent` on leases / rent_snapshots.
  - DO NOT join leases directly to rent_charge_lines and then LIMIT — leases
    has one row per unit but rent_charge_lines has MANY rows per unit (one
    per charge_code per month). A naive JOIN explodes; LIMIT 2 then returns
    the same unit twice. Always aggregate rent_charge_lines first.

Worked patterns (copy these idioms verbatim when applicable):

  -- Compare N distinct units on rent + move-out + a specific fee.
  -- The fee subquery aggregates rent_charge_lines to ONE row per unit
  -- before joining, so LIMIT N returns N distinct units.
  WITH amenity AS (
    SELECT unit_number,
           SUM(amount) AS amenity_fee_latest_month
    FROM rent_charge_lines
    WHERE charge_code = 'AMENITY'
      AND snapshot_month = (
        SELECT MAX(snapshot_month) FROM rent_charge_lines
      )
    GROUP BY unit_number
  )
  SELECT l.unit_number,
         l.monthly_rent,
         l.move_out_date,
         COALESCE(a.amenity_fee_latest_month, 0) AS amenity_fee
  FROM leases l
  LEFT JOIN amenity a ON a.unit_number = l.unit_number
  ORDER BY l.unit_number
  LIMIT 2;

  -- All charge-code totals for a unit, one row per code:
  SELECT charge_code, SUM(amount) AS total
  FROM rent_charge_lines
  WHERE unit_number = 'A103'
    AND snapshot_month = (SELECT MAX(snapshot_month) FROM rent_charge_lines)
  GROUP BY charge_code
  ORDER BY total DESC;

  -- Move-outs in a date range:
  SELECT unit_number, tenant_id, move_out_date, monthly_rent
  FROM leases
  WHERE move_out_date BETWEEN '2025-10-01' AND '2026-03-31'
  ORDER BY move_out_date;"""
        return SQL_TOOLS["execute_scoped_sql"]["fn"](scope_codes, sql)

    @tool
    def search_property_pages(
        query: str,
        k: int = 4,
        max_images: int = 3,
        property_code: str | None = None,
    ) -> dict:
        """Semantic + image search over the active property's MARKETING WEBSITE chunks (amenities, floor plans, gallery, pet policy, etc). Returns text chunks AND a multimodal `images` list — relevant property photos are then surfaced as a gallery beneath your reply automatically. Use for qualitative questions and any user request to 'see/show/display' something visual. `max_images` caps the gallery size (default 3, hard ceiling 25) — set HIGH (15-25) when the user asks 'show me ALL the images / every photo / show them all'. In compare mode pass `property_code` to pick which property."""
        c = _pick_code(property_code)
        if isinstance(c, dict): return c
        return search_property_active(c, query, k=k, max_images=max_images)

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
        get_expiring_leases, get_top_balances, get_lease_deposits, get_move_outs,
        get_unit_charges, compare_units, list_units,
        # compare_properties,  # DISABLED — cross-property compare removed (see @tool block above)
        execute_scoped_sql, search_property_pages, render_chart,
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

def _scope_summary(scope: dict, time_scope: dict | None = None) -> str:
    kind = scope.get("kind")
    if kind == "single":
        base = f"single property {scope.get('code')!r}"
    elif kind == "compare":
        base = f"comparison across properties {scope.get('codes')}"
    else:
        base = f"unresolved ({kind})"
    # Include time scope so drift detection in enter_turn fires when the
    # user picks a different month on a subsequent turn.
    if time_scope:
        tk = time_scope.get("kind")
        tm = time_scope.get("month")
        if tk == "specific" and tm:
            base += f"; time=snapshot_month={tm}"
        elif tk == "latest":
            base += "; time=latest"
    return base


def _build_system_prompt(state: ChatState) -> str:
    """One prompt, built once per request from scope + property_name + time_scope."""
    scope = state.get("scope") or {}
    scope_summary = _scope_summary(scope)
    time_scope = state.get("time_scope") or {}

    base = (
        "You are a property-management AI analyst. You answer using the "
        "bound tools — do not invent numbers.\n\n"
        f"Active scope: {scope_summary}.\n"
    )
    if scope.get("kind") == "single" and state.get("property_name"):
        base += f"Active property name: {state['property_name']}.\n"

    # ---- HARD REFUSAL: stay in domain + stay in scope ---------------------
    # The agent must refuse two classes of question outright:
    #   (a) Anything not about property management for the active property
    #       (general knowledge, math, coding, weather, jokes, etc.)
    #   (b) Questions about a DIFFERENT property than the active one.
    # The clarify node already catches dropdown/message disagreements, but
    # if a user asks "what about <other property>?" in a follow-up turn the
    # graph won't re-clarify — this prompt rule covers that case.
    _refusal_target = state.get("property_name") or scope.get("code") or "the active property"
    _active_codes = (
        [scope.get("code")] if scope.get("kind") == "single" and scope.get("code")
        else (scope.get("codes") or [])
    )
    base += (
        "\nHARD REFUSAL RULES — non-negotiable:\n"
        f"  - You ONLY answer questions about {_refusal_target}"
        f" (property code(s): {_active_codes}).\n"
        "  - If the user asks about ANY other property, or about a topic "
        "unrelated to this property's rent roll / units / leases / residents "
        "/ charges / amenities / floor plans / marketing site, you MUST "
        "reply with EXACTLY this one sentence and call no tools:\n"
        f"      I can only answer questions about {_refusal_target}. "
        "Please rephrase your question or switch the active property in the dropdown.\n"
        "  - Do not be 'helpful' by partially answering off-scope questions, "
        "do not provide world knowledge, do not write code, do not do math "
        "puzzles. Refuse and stop.\n"
        "  - Greetings ('hi', 'thanks') are fine — respond briefly and "
        "invite a property question.\n"
    )
    # -----------------------------------------------------------------------

    # DISABLED: cross-property compare mode (kept commented out so the code is
    # easy to restore by uncommenting). The frontend no longer lets users
    # select multiple properties, so scope.kind == "compare" should not occur
    # via the UI path. Free-text "compare 115r and 134r" requests via the
    # clarify dropdown are still parsed by guardrails/scope.py but will leave
    # the agent without a compare_properties tool — it'll respond per-property.
    # if scope.get("kind") == "compare":
    #     codes = scope.get("codes") or []
    #     base += (
    #         f"Compare mode is active across {len(codes)} properties: {', '.join(codes)}.\n"
    #         f"  - Single-property tools (list_units, get_property_summary, get_unit_mix, get_occupancy,\n"
    #         f"    get_rent_trend, get_expiring_leases, get_top_balances, get_unit_charges,\n"
    #         f"    compare_units, search_property_pages) accept an optional `property_code` arg.\n"
    #         f"  - When the user asks about ONE of the compare codes, pass property_code=<that_code>.\n"
    #         f"  - When the user asks to compare 'across all' or names two codes, CALL THE TOOL "
    #         f"ONCE PER property_code in scope (e.g. list_units(max_rent=2000, property_code='115r') "
    #         f"AND list_units(max_rent=2000, property_code='134r') in the SAME turn — emit BOTH tool calls in parallel).\n"
    #         f"  - For a single aggregate metric across all properties, use compare_properties.\n"
    #     )

    # Time-scope: the user already told us (or was asked) "as of which month".
    tk = (time_scope or {}).get("kind")
    tm = (time_scope or {}).get("month")
    tl = (time_scope or {}).get("label")
    if tk == "specific" and tm:
        base += (
            f"\nTime scope: {tl} (snapshot_month='{tm}').\n"
            f"  - Pass snapshot_month='{tm}' to EVERY tool call that accepts it "
            f"(get_property_summary, get_top_balances, get_lease_deposits, "
            f"get_occupancy, get_unit_charges, get_rent_trend).\n"
            f"  - For tools without a snapshot_month arg (list_units, compare_units, "
            f"get_unit_mix, get_expiring_leases, get_move_outs), use "
            f"execute_scoped_sql against rent_snapshots with snapshot_month='{tm}'.\n"
            f"  - Always state the snapshot month explicitly in your reply.\n"
        )
    elif tk == "latest":
        base += (
            "\nTime scope: LATEST snapshot.\n"
            "  - Default tool behaviour returns the latest snapshot — no need "
            "to pass snapshot_month.\n"
            "  - Always say 'as of the latest snapshot' (or the actual month name "
            "if a tool returns one) in your reply.\n"
        )
    elif tk == "any":
        base += (
            "\nTime scope: not applicable (this question doesn't depend on a "
            "specific month — e.g. amenities, photos, property metadata).\n"
        )

    base += (
        "\nTool selection guide (use the MOST specific):\n"
        "  - Specific unit fees / charges ('parking for A103')      → get_unit_charges\n"
        "  - Compare 2+ units in same property                       → compare_units\n"
        # DISABLED: cross-property compare removed.
        # "  - Compare 2+ properties on a metric                       → compare_properties\n"
        "  - 'average rent / occupancy / summary'                    → get_property_summary\n"
        "  - Unit-type breakdown                                     → get_unit_mix\n"
        "  - Monthly trend / over the year                           → get_rent_trend\n"
        "  - Expiring leases                                         → get_expiring_leases\n"
        "  - High balances / delinquent                              → get_top_balances\n"
        "  - Deposits (resident / other) — sum, avg, largest         → get_lease_deposits\n"
        "  - Move-outs — who's leaving, when                         → get_move_outs\n"
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
        "\nImages / photos — read carefully:\n"
        "  - The UI renders relevant property photos as a gallery beneath your reply "
        "automatically whenever search_property_pages is called. The user CAN see them.\n"
        "  - If the user asks to 'show', 'display', 'see', or otherwise reference an "
        "image / photo / picture / view / look — you MUST call search_property_pages "
        "with a query that captures what they want to see (e.g. 'pool', 'kitchen', "
        "'river view'). This applies on FOLLOW-UP turns too — re-call the tool, don't "
        "assume images from a previous turn are still on screen.\n"
        "  - **Honesty about captions:** describe each surfaced image USING ITS ACTUAL "
        "CAPTION from the tool's `images` list. Do NOT relabel images to fit the "
        "user's question. If the user asked for 'floor plans' but the returned captions "
        "are lifestyle photos (no caption mentions 'floor plan' / 'layout' / 'diagram' / "
        "'bedroom plan'), TELL THE USER: 'the marketing site doesn't host static "
        "floor-plan diagrams — here's what IS available from the /floorplans/ page'. "
        "Same for any specific request that's not actually present.\n"
        "  - **'All / every / show them all' intent:** when the user says 'show me ALL "
        "the images / every photo / show them all', pass max_images=20 (or 25) to "
        "search_property_pages so the gallery isn't capped at 3.\n"
        "  - **Different question, fresh call:** if the user asks for floor plans after "
        "asking for amenities, do NOT regurgitate the previous answer — call "
        "search_property_pages again with the new query so the gallery refreshes.\n"
        "  - NEVER say \"I can't display images\". Calling search_property_pages IS how "
        "images get shown. NEVER embed `![](...)` markdown image links — they will be "
        "stripped from your reply. Talk about the images briefly, then let the gallery "
        "do the work.\n"
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
        "\nSnapshot-month transparency:\n"
        "  - The rent-roll holds 12 monthly snapshots per property. Charges, "
        "rent, and balances all CHANGE month-to-month.\n"
        "  - Every numeric answer derived from get_unit_charges, "
        "get_property_summary, list_units, etc. MUST state the snapshot month "
        "it covers (e.g. \"As of the December 2025 snapshot, unit A103 paid "
        "$15 for trash\").\n"
        "  - When a user asks 'how much for X?' without specifying a month, "
        "answer with the LATEST snapshot's value AND name it. Offer the historical "
        "trend if useful (a brief 'this fee was $25 earlier in the year, dropped to "
        "$15 in August').\n"
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

    # streaming=True is required for LangGraph's `messages` stream mode to
    # surface token chunks. The sync .invoke() path still aggregates them
    # at the call site, so non-streaming callers see the same final result.
    if provider == "openai":
        if not _settings.openai_api_key:
            raise ProviderUnavailable("OPENAI_API_KEY not set")
        return ChatOpenAI(
            model=model, temperature=temperature,
            api_key=_settings.openai_api_key,
            streaming=True,
        )

    if provider == "anthropic":
        if not _settings.anthropic_api_key:
            raise ProviderUnavailable("ANTHROPIC_API_KEY not set")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model, temperature=temperature,
            api_key=_settings.anthropic_api_key,
            streaming=True,
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

    # v5: also classify the time-scope of the question.
    from ..guardrails.scope import extract_time_intent
    out["time_scope"] = extract_time_intent(state.get("user_message", ""))
    return out


def scope_router(state: ChatState) -> str:
    """Route after extract_scope. Property scope is the first gate; time
    scope is the second. Either missing → clarify."""
    kind = (state.get("scope") or {}).get("kind", "missing")
    if kind in {"conflict", "missing"}:
        return "clarify"
    time_kind = (state.get("time_scope") or {}).get("kind", "any")
    if time_kind == "missing":
        return "clarify_time"
    return "enter_turn"


# ---------------------------------------------------------------------------
# 2. clarify — LangGraph interrupt; resumes here with the user's reply
# ---------------------------------------------------------------------------

def clarify(state: ChatState) -> dict[str, Any]:
    scope = state.get("scope") or {}
    kind = scope.get("kind")

    # Single-property mode only. `conflict` means the user mentioned a code
    # that disagrees with the dropdown, OR mentioned 2+ codes with no
    # dropdown set. Ask which ONE to use — never auto-merge.
    if kind == "conflict":
        q_code = scope.get("query_code")
        d_code = scope.get("dropdown_code")
        if d_code:
            question = (
                f"You're scoped to **{d_code}** but your message mentioned "
                f"**{q_code}**. I only answer about one property at a time — "
                f"which should I use for this turn?"
            )
            options = [d_code, q_code]
        else:
            available = scope.get("available") or [q_code]
            question = (
                f"Your message mentioned multiple properties "
                f"({', '.join(available)}). I only answer about one property "
                f"at a time — which one did you mean?"
            )
            options = available
    elif kind == "missing":
        question = (
            "Which property are you asking about? I didn't find a property "
            "code in your message and none is selected in the dropdown."
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
        # Accept compound phrases like "compare 115r and 134r" by extracting
        # all property-code tokens from the reply.
        from ..guardrails.scope import extract_codes_from_message
        codes_in_reply = extract_codes_from_message(user_choice)
        if codes_in_reply:
            chosen = codes_in_reply
        else:
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

    # Single-property only — if the user's reply somehow contained multiple
    # valid codes, take the FIRST and ignore the rest (compare mode is off).
    if valid:
        chosen_code = valid[0]
        _, name = validate_property_code(chosen_code)
        return {
            "scope": ScopeDecision(kind="single", code=chosen_code, source="resumed").to_dict(),
            "property_name": name,
        }

    from ..guardrails.scope import _all_property_codes
    return {"scope": ScopeDecision(kind="missing", available=_all_property_codes()).to_dict()}


# ---------------------------------------------------------------------------
# 2b. clarify_time — second interrupt for missing month
# ---------------------------------------------------------------------------

def clarify_time(state: ChatState) -> dict[str, Any]:
    """Ask the user which snapshot month they meant. Fires only when the
    question is time-sensitive AND no month / "latest" intent was given."""
    from ..guardrails.scope import available_snapshot_months, extract_time_intent

    months = available_snapshot_months()  # newest first, ISO strings
    # Friendly labels for the buttons (e.g. "December 2025").
    from datetime import date as _date
    def label(iso: str) -> str:
        y, m, _ = iso.split("-")
        return _date(int(y), int(m), 1).strftime("%B %Y")

    options = ["Latest"] + [label(m) for m in months]

    question = (
        "Which month should I use for this answer? "
        "Rent, charges, occupancy and balances all change month-to-month — "
        "pick a specific snapshot, or 'Latest' for the most recent one."
    )

    user_choice = interrupt({
        "question": question,
        "options": options,
        "scope_kind": "time",
    })

    raw = user_choice if isinstance(user_choice, str) else (
        user_choice[0] if isinstance(user_choice, list) and user_choice else ""
    )
    raw_l = (raw or "").strip().lower()

    # Latest → mark as latest.
    if raw_l in {"latest", "current", "newest", "most recent", "recent"}:
        return {"time_scope": {"kind": "latest", "month": None, "label": "latest snapshot"}}

    # Try to parse a specific month from the reply.
    parsed = extract_time_intent(raw)
    if parsed["kind"] == "specific":
        return {"time_scope": parsed}
    if parsed["kind"] == "latest":
        return {"time_scope": parsed}

    # Last-ditch: match raw against the available labels.
    for iso in months:
        if label(iso).lower() == raw_l or iso == raw_l:
            return {"time_scope": {"kind": "specific", "month": iso, "label": label(iso)}}

    # Unparseable reply → re-loop the interrupt.
    return {"time_scope": {"kind": "missing", "month": None, "label": None}}


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


def _orphaned_tool_repairs(messages: list) -> list[ToolMessage]:
    """Build stub ToolMessages for any unresolved tool_calls in conversation history.

    When the user clicks Stop in the UI after the agent emitted an AIMessage
    with tool_calls but BEFORE the tools node executed, those tool_call_ids
    have no corresponding ToolMessage. On the next /chat turn the OpenAI
    Chat Completions API rejects the conversation with:

        BadRequestError 400 — "An assistant message with 'tool_calls' must
        be followed by tool messages responding to each 'tool_call_id'."

    To stay valid without nuking the conversation, walk back to the most
    recent AIMessage that has tool_calls and emit a stub ToolMessage for
    every tool_call_id that doesn't already have a response further down.
    """
    if not messages:
        return []
    # Locate the most-recent AIMessage that requested tools.
    last_ai_with_calls = None
    last_ai_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_ai_with_calls = m
            last_ai_idx = i
            break
    if last_ai_with_calls is None:
        return []

    # Collect every ToolMessage that landed AFTER that AIMessage, keyed by
    # tool_call_id, so we know which calls have been resolved.
    answered: set[str] = set()
    for m in messages[last_ai_idx + 1:]:
        if isinstance(m, ToolMessage) and m.tool_call_id:
            answered.add(m.tool_call_id)

    repairs: list[ToolMessage] = []
    for tc in last_ai_with_calls.tool_calls:
        tcid = tc.get("id")
        if tcid and tcid not in answered:
            repairs.append(ToolMessage(
                tool_call_id=tcid,
                name=tc.get("name", "unknown"),
                content="(cancelled — user stopped generation before this tool ran)",
            ))
    return repairs


def enter_turn(state: ChatState) -> dict[str, Any]:
    """Open a new conversational turn.

    - First turn: seed the full SystemMessage + the user's HumanMessage.
    - Later turns: append only the new HumanMessage. If the active scope
      drifted from the prior turn (different property), append a small
      'scope refresh' SystemMessage so the LLM knows.
    - Resets per-turn state regardless.
    """
    existing = state.get("messages") or []
    current_summary = _scope_summary(
        state.get("scope") or {}, state.get("time_scope") or {}
    )
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
        # Property OR time scope drifted between turns — append a refresh
        # note rather than a full re-prompt. Positioned at the end so it's
        # the most-recent instruction the LLM sees.
        refresh_lines = [
            f"{_SCOPE_MARKER} {current_summary}",
            "",
            f"Scope updated for this turn: {current_summary}.",
            "Use ONLY this scope when answering the next user message.",
        ]
        # Re-emphasise the time scope so the LLM doesn't carry over the
        # previous turn's month or hallucinate a different one.
        tk = (state.get("time_scope") or {}).get("kind")
        tm = (state.get("time_scope") or {}).get("month")
        tl = (state.get("time_scope") or {}).get("label")
        if tk == "specific" and tm:
            refresh_lines.append(
                f"TIME SCOPE: {tl} (snapshot_month='{tm}'). You MUST call a tool "
                f"with snapshot_month='{tm}' to answer — do not infer the answer "
                f"from prior conversation history or use a different month."
            )
        elif tk == "latest":
            refresh_lines.append(
                "TIME SCOPE: LATEST snapshot. Call a tool with no snapshot_month "
                "argument (tools default to latest)."
            )
        new_msgs.append(SystemMessage(content="\n".join(refresh_lines)))

    # Repair any orphaned tool_calls left over from a Stop-aborted prior
    # turn so OpenAI doesn't reject this conversation on submission.
    new_msgs.extend(_orphaned_tool_repairs(existing))

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


# Strip ALL inline images from the markdown body. RAG v2 surfaces images as
# proper `image` UI components rendered as a gallery beneath the message, so
# the LLM should never embed `![](...)` in the prose. We strip even legit
# external URLs since they're either (a) duplicates of the gallery or (b)
# the LLM hallucinating a working URL out of the captions it saw.
_INLINE_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)", re.IGNORECASE | re.DOTALL)


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

    # Sources + images: from the last successful RAG step (if any).
    sources: list[dict] = []
    last_rag = next(
        (s for s in reversed(history)
         if s.get("ok") and s.get("tool") == "search_property_pages"),
        None,
    )
    if last_rag and isinstance(last_rag.get("result"), dict):
        rag_result = last_rag["result"]
        sources = rag_result.get("sources") or []
        # v2 returns relevant images alongside text chunks — surface them as
        # `image` UIComponents so the frontend renders them inline.
        for img in (rag_result.get("images") or []):
            if not isinstance(img, dict) or not img.get("url"):
                continue
            components.append({
                "type": "image",
                "title": img.get("caption") or img.get("section_path") or "Image",
                "data": {
                    "src": img["url"],
                    "caption": img.get("caption") or "",
                    "source_url": img.get("source_url") or "",
                },
            })

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
