"""SQL analytical tools.

Six read-only functions over the rent-roll MySQL data. Every function:

  1. Takes `property_code` as a required first argument.
  2. Calls `require_scope(property_code)` to assert non-empty scope.
  3. Issues a parameterised SQL query with an explicit
     `WHERE property_code = :code` clause.

The LLM never writes SQL. It picks one of these functions and supplies the
arguments. This is the assignment's strongest scoping guarantee at the data
layer — there is no path from the model to the database that bypasses the
property_code filter.

Each function returns a plain dict — the response composer turns those into
Markdown answers and UI components.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import text

from ..db import session_scope
from ..guardrails.scope import require_scope


# ---------------------------------------------------------------------------
# Tool registry — used by the LangGraph SQL node and the FastAPI debug route.
# ---------------------------------------------------------------------------

def _rows_to_dicts(result) -> list[dict[str, Any]]:
    keys = result.keys()
    return [dict(zip(keys, row)) for row in result.fetchall()]


def _stringify_dates(d: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly: convert date/datetime values to ISO strings."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (date, datetime)):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# 1. Property summary
# ---------------------------------------------------------------------------

def get_property_summary(property_code: str) -> dict[str, Any]:
    """High-level KPIs for the property's most recent monthly snapshot."""
    code = require_scope(property_code)
    with session_scope() as s:
        prop = s.execute(
            text("""
                SELECT property_code, property_name, property_type
                FROM properties
                WHERE property_code = :code
            """),
            {"code": code},
        ).first()
        if prop is None:
            return {"property_code": code, "found": False}

        latest_month = s.execute(
            text("""
                SELECT MAX(snapshot_month) FROM rent_snapshots
                WHERE property_code = :code
            """),
            {"code": code},
        ).scalar()

        if latest_month is None:
            return {
                "property_code": prop.property_code,
                "property_name": prop.property_name,
                "property_type": prop.property_type,
                "found": True,
                "has_data": False,
                "note": "No rent-roll snapshots available for this property.",
            }

        agg = s.execute(
            text("""
                SELECT
                    COUNT(*)                                              AS total_units,
                    SUM(CASE WHEN occupied THEN 1 ELSE 0 END)              AS occupied_units,
                    AVG(CASE WHEN monthly_rent > 0 THEN monthly_rent END)  AS avg_rent,
                    SUM(CASE WHEN monthly_rent > 0 THEN monthly_rent END)  AS rent_roll_total
                FROM rent_snapshots
                WHERE property_code = :code
                  AND snapshot_month = :month
            """),
            {"code": code, "month": latest_month},
        ).first()

    return {
        "property_code": prop.property_code,
        "property_name": prop.property_name,
        "property_type": prop.property_type,
        "found": True,
        "has_data": True,
        "snapshot_month": latest_month.isoformat() if latest_month else None,
        "total_units": int(agg.total_units or 0),
        "occupied_units": int(agg.occupied_units or 0),
        "occupancy_pct": (
            round(100 * agg.occupied_units / agg.total_units, 1)
            if agg.total_units else None
        ),
        "avg_rent": round(float(agg.avg_rent), 2) if agg.avg_rent else None,
        "rent_roll_total": round(float(agg.rent_roll_total), 2) if agg.rent_roll_total else None,
    }


# ---------------------------------------------------------------------------
# 2. Unit mix
# ---------------------------------------------------------------------------

def get_unit_mix(property_code: str) -> dict[str, Any]:
    """Breakdown by unit_type: count, avg market rent, avg sqft.

    Sourced from `units` (latest snapshot) so it reflects the current mix.
    """
    code = require_scope(property_code)
    with session_scope() as s:
        rows = _rows_to_dicts(s.execute(
            text("""
                SELECT
                    COALESCE(unit_type, 'unspecified') AS unit_type,
                    COUNT(*)                            AS unit_count,
                    ROUND(AVG(market_rent), 0)          AS avg_market_rent,
                    ROUND(AVG(sqft), 0)                 AS avg_sqft
                FROM units
                WHERE property_code = :code
                GROUP BY unit_type
                ORDER BY unit_count DESC
            """),
            {"code": code},
        ))
    return {
        "property_code": code,
        "row_count": len(rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. Occupancy (single month, defaults to latest)
# ---------------------------------------------------------------------------

def get_occupancy(property_code: str, month: str | None = None) -> dict[str, Any]:
    """Occupancy % for a given month. `month` is 'YYYY-MM' or None (latest)."""
    code = require_scope(property_code)
    with session_scope() as s:
        if month:
            try:
                y, m = month.split("-")
                month_date = date(int(y), int(m), 1)
            except (ValueError, AttributeError):
                return {"property_code": code, "error": f"Invalid month: {month!r}"}
        else:
            month_date = s.execute(
                text("SELECT MAX(snapshot_month) FROM rent_snapshots WHERE property_code = :code"),
                {"code": code},
            ).scalar()
            if month_date is None:
                return {"property_code": code, "found": False, "note": "No snapshots."}

        agg = s.execute(
            text("""
                SELECT
                    COUNT(*)                                  AS total_units,
                    SUM(CASE WHEN occupied THEN 1 ELSE 0 END) AS occupied_units
                FROM rent_snapshots
                WHERE property_code = :code
                  AND snapshot_month = :month
            """),
            {"code": code, "month": month_date},
        ).first()

    total = int(agg.total_units or 0)
    occ = int(agg.occupied_units or 0)
    return {
        "property_code": code,
        "snapshot_month": month_date.isoformat() if month_date else None,
        "total_units": total,
        "occupied_units": occ,
        "vacant_units": total - occ,
        "occupancy_pct": round(100 * occ / total, 1) if total else None,
    }


# ---------------------------------------------------------------------------
# 4. Rent trend (time series across snapshot months)
# ---------------------------------------------------------------------------

def get_rent_trend(property_code: str, months: int = 12) -> dict[str, Any]:
    """Monthly avg-rent and occupancy series over the last N months."""
    code = require_scope(property_code)
    months = max(1, min(int(months or 12), 36))
    with session_scope() as s:
        rows = _rows_to_dicts(s.execute(
            text("""
                SELECT
                    snapshot_month,
                    COUNT(*)                                              AS total_units,
                    SUM(CASE WHEN occupied THEN 1 ELSE 0 END)             AS occupied_units,
                    ROUND(AVG(CASE WHEN monthly_rent > 0 THEN monthly_rent END), 0) AS avg_rent
                FROM rent_snapshots
                WHERE property_code = :code
                GROUP BY snapshot_month
                ORDER BY snapshot_month DESC
                LIMIT :limit
            """),
            {"code": code, "limit": months},
        ))
    # Reverse so chart x-axis goes oldest -> newest.
    rows.reverse()
    series = [
        {
            "month": (r["snapshot_month"].isoformat()
                      if isinstance(r["snapshot_month"], (date, datetime)) else r["snapshot_month"]),
            "avg_rent": float(r["avg_rent"]) if r["avg_rent"] is not None else None,
            "occupancy_pct": (round(100 * r["occupied_units"] / r["total_units"], 1)
                              if r["total_units"] else None),
            "total_units": int(r["total_units"]),
            "occupied_units": int(r["occupied_units"]),
        }
        for r in rows
    ]
    return {
        "property_code": code,
        "months": len(series),
        "series": series,
    }


# ---------------------------------------------------------------------------
# 5. Expiring leases
# ---------------------------------------------------------------------------

def get_expiring_leases(
    property_code: str,
    within_days: int = 90,
    reference_date: str | None = None,
) -> dict[str, Any]:
    """Leases expiring within N days of `reference_date` (default: today)."""
    code = require_scope(property_code)
    within_days = max(1, min(int(within_days or 90), 365))

    if reference_date:
        try:
            ref = datetime.strptime(reference_date, "%Y-%m-%d").date()
        except ValueError:
            return {"property_code": code, "error": f"Invalid reference_date: {reference_date!r}"}
    else:
        ref = date.today()

    with session_scope() as s:
        rows = _rows_to_dicts(s.execute(
            text("""
                SELECT
                    unit_number,
                    tenant_id,
                    lease_start,
                    lease_end,
                    monthly_rent,
                    balance,
                    DATEDIFF(lease_end, :ref) AS days_until_expiry
                FROM leases
                WHERE property_code = :code
                  AND lease_end IS NOT NULL
                  AND lease_end >= :ref
                  AND lease_end <= DATE_ADD(:ref, INTERVAL :days DAY)
                ORDER BY lease_end ASC
                LIMIT 100
            """),
            {"code": code, "ref": ref, "days": within_days},
        ))

    rows = [_stringify_dates(r) for r in rows]
    return {
        "property_code": code,
        "reference_date": ref.isoformat(),
        "within_days": within_days,
        "row_count": len(rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 6. Top outstanding balances
# ---------------------------------------------------------------------------

def get_top_balances(property_code: str, n: int = 10) -> dict[str, Any]:
    """Top N leases by outstanding balance (most-owed first)."""
    code = require_scope(property_code)
    n = max(1, min(int(n or 10), 50))
    with session_scope() as s:
        rows = _rows_to_dicts(s.execute(
            text("""
                SELECT
                    unit_number,
                    tenant_id,
                    monthly_rent,
                    balance,
                    lease_end,
                    status
                FROM leases
                WHERE property_code = :code
                  AND balance IS NOT NULL
                ORDER BY balance DESC
                LIMIT :n
            """),
            {"code": code, "n": n},
        ))
    rows = [_stringify_dates(r) for r in rows]
    return {
        "property_code": code,
        "row_count": len(rows),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Registry — consumed by the LangGraph SQL node.
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "get_property_summary": {
        "fn": get_property_summary,
        "description": "Headline KPIs for the latest snapshot: unit count, occupancy %, avg rent, total rent roll.",
        "params": [],
    },
    "get_unit_mix": {
        "fn": get_unit_mix,
        "description": "Breakdown by unit_type (count, avg market rent, avg sqft).",
        "params": [],
    },
    "get_occupancy": {
        "fn": get_occupancy,
        "description": "Occupancy % for a single month (default: latest).",
        "params": ["month?"],
    },
    "get_rent_trend": {
        "fn": get_rent_trend,
        "description": "Monthly avg-rent and occupancy time series.",
        "params": ["months?"],
    },
    "get_expiring_leases": {
        "fn": get_expiring_leases,
        "description": "Leases expiring within N days (default 90).",
        "params": ["within_days?", "reference_date?"],
    },
    "get_top_balances": {
        "fn": get_top_balances,
        "description": "Leases with highest outstanding balance (default top 10).",
        "params": ["n?"],
    },
}


def run_tool(name: str, property_code: str, **kwargs: Any) -> dict[str, Any]:
    """Look up and run a tool by name. Scope enforcement happens inside."""
    if name not in TOOLS:
        raise KeyError(f"Unknown SQL tool: {name}")
    return TOOLS[name]["fn"](property_code, **kwargs)
