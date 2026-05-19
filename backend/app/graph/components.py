"""Deterministic UI-component builder.

The LLM writes the markdown narrative, but UI components are emitted by
THIS module — based purely on the SQL tool result shape. That keeps the
charts/tables consistent regardless of model quirks.

Each builder returns a list of component dicts shaped to match
`schemas.UIComponent`:

    {"type": "kpi" | "table" | "bar_chart" | "line_chart",
     "title": str,
     "data": {...}}
"""
from __future__ import annotations

from typing import Any


def _fmt_money(v: float | int | None) -> str:
    if v is None:
        return "—"
    return f"${v:,.0f}"


def _fmt_pct(v: float | int | None) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}%"


# ---------------------------------------------------------------------------
# Per-tool builders
# ---------------------------------------------------------------------------

def from_property_summary(r: dict[str, Any]) -> list[dict]:
    if not r.get("has_data"):
        return []
    return [
        {
            "type": "kpi",
            "title": "Occupancy",
            "data": {
                "value": _fmt_pct(r.get("occupancy_pct")),
                "subtitle": f"{r.get('occupied_units')} of {r.get('total_units')} units",
                "month": r.get("snapshot_month"),
            },
        },
        {
            "type": "kpi",
            "title": "Avg Monthly Rent",
            "data": {
                "value": _fmt_money(r.get("avg_rent")),
                "subtitle": "per occupied unit",
                "month": r.get("snapshot_month"),
            },
        },
        {
            "type": "kpi",
            "title": "Total Rent Roll",
            "data": {
                "value": _fmt_money(r.get("rent_roll_total")),
                "subtitle": "current monthly",
                "month": r.get("snapshot_month"),
            },
        },
    ]


def from_unit_mix(r: dict[str, Any]) -> list[dict]:
    rows = r.get("rows") or []
    if not rows:
        return []
    return [{
        "type": "table",
        "title": "Unit Mix",
        "data": {
            "columns": ["Unit Type", "Count", "Avg Market Rent", "Avg Sqft"],
            "rows": [
                [row["unit_type"], row["unit_count"],
                 _fmt_money(row.get("avg_market_rent")), row.get("avg_sqft")]
                for row in rows
            ],
        },
    }]


def from_occupancy(r: dict[str, Any]) -> list[dict]:
    if r.get("total_units") is None:
        return []
    return [{
        "type": "kpi",
        "title": "Occupancy",
        "data": {
            "value": _fmt_pct(r.get("occupancy_pct")),
            "subtitle": f"{r.get('occupied_units')} occupied · {r.get('vacant_units')} vacant",
            "month": r.get("snapshot_month"),
        },
    }]


def from_rent_trend(r: dict[str, Any]) -> list[dict]:
    series = r.get("series") or []
    if not series:
        return []
    return [{
        "type": "line_chart",
        "title": f"Avg Rent — Last {len(series)} Months",
        "data": {
            "x_label": "Month",
            "y_label": "Avg Rent ($)",
            "x": [s["month"][:7] for s in series],
            "y": [s["avg_rent"] for s in series],
            "secondary": {
                "label": "Occupancy %",
                "y": [s["occupancy_pct"] for s in series],
            },
        },
    }]


def from_expiring_leases(r: dict[str, Any]) -> list[dict]:
    rows = r.get("rows") or []
    if not rows:
        return []
    return [{
        "type": "table",
        "title": f"Leases Expiring in {r.get('within_days')} days (from {r.get('reference_date')})",
        "data": {
            "columns": ["Unit", "Tenant", "Lease End", "Days Left", "Rent", "Balance"],
            "rows": [
                [row["unit_number"], row.get("tenant_id"),
                 row.get("lease_end"), row.get("days_until_expiry"),
                 _fmt_money(row.get("monthly_rent")),
                 _fmt_money(row.get("balance"))]
                for row in rows
            ],
        },
    }]


def from_top_balances(r: dict[str, Any]) -> list[dict]:
    rows = r.get("rows") or []
    if not rows:
        return []
    return [{
        "type": "table",
        "title": f"Top {len(rows)} Outstanding Balances",
        "data": {
            "columns": ["Unit", "Tenant", "Balance", "Monthly Rent", "Lease End"],
            "rows": [
                [row["unit_number"], row.get("tenant_id"),
                 _fmt_money(row.get("balance")),
                 _fmt_money(row.get("monthly_rent")),
                 row.get("lease_end")]
                for row in rows
            ],
        },
    }]


# ---------------------------------------------------------------------------
# Registry — keyed by SQL tool name
# ---------------------------------------------------------------------------

BUILDERS = {
    "get_property_summary": from_property_summary,
    "get_unit_mix":         from_unit_mix,
    "get_occupancy":        from_occupancy,
    "get_rent_trend":       from_rent_trend,
    "get_expiring_leases":  from_expiring_leases,
    "get_top_balances":     from_top_balances,
}


def build_components(tool_name: str | None, result: dict | None) -> list[dict]:
    if not tool_name or not result:
        return []
    fn = BUILDERS.get(tool_name)
    if not fn:
        return []
    try:
        return fn(result)
    except Exception:
        return []
