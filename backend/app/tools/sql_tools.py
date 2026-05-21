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
# 7. Unit charges — every line item for one unit (preserves multiplicity)
# ---------------------------------------------------------------------------

def get_unit_charges(
    property_code: str,
    unit_number: str,
    snapshot_month: str | None = None,
) -> dict[str, Any]:
    """List every charge line for a unit in source order.

    Unlike `raw_row.charges` (which sums by code), this surfaces each row
    distinctly — so two PARKING lines of $75 and $100 appear as two entries.
    """
    code = require_scope(property_code)
    if not unit_number:
        return {"property_code": code, "error": "unit_number is required"}

    with session_scope() as s:
        if snapshot_month:
            try:
                y, m = snapshot_month.split("-")[:2]
                month_date = date(int(y), int(m), 1)
            except (ValueError, AttributeError):
                return {"property_code": code, "error": f"Invalid snapshot_month: {snapshot_month!r}"}
        else:
            month_date = s.execute(
                text("""
                    SELECT MAX(snapshot_month) FROM rent_charge_lines
                    WHERE property_code = :code AND unit_number = :unit
                """),
                {"code": code, "unit": unit_number},
            ).scalar()
            if month_date is None:
                return {
                    "property_code": code, "unit_number": unit_number,
                    "found": False,
                    "note": f"No charge lines for unit {unit_number} in {code}.",
                }

        rows = _rows_to_dicts(s.execute(
            text("""
                SELECT line_index, charge_code, amount
                FROM rent_charge_lines
                WHERE property_code = :code
                  AND unit_number = :unit
                  AND snapshot_month = :month
                ORDER BY line_index
            """),
            {"code": code, "unit": unit_number, "month": month_date},
        ))

        # Roll up: how many lines per code? Helps the LLM say
        # "there are 2 PARKING lines: $75 and $100".
        per_code: dict[str, list[float]] = {}
        for r in rows:
            per_code.setdefault(r["charge_code"], []).append(float(r["amount"]) if r["amount"] is not None else 0.0)
        summary = [
            {"charge_code": code_name, "count": len(amts),
             "amounts": amts, "total": round(sum(amts), 2)}
            for code_name, amts in sorted(per_code.items(), key=lambda kv: -sum(kv[1]))
        ]

    return {
        "property_code": code,
        "unit_number": unit_number,
        "snapshot_month": month_date.isoformat() if month_date else None,
        "found": bool(rows),
        "line_count": len(rows),
        "lines": [
            {"line_index": r["line_index"],
             "charge_code": r["charge_code"],
             "amount": float(r["amount"]) if r["amount"] is not None else None}
            for r in rows
        ],
        "summary_by_code": summary,
    }


# ---------------------------------------------------------------------------
# 8. Compare units within one property
# ---------------------------------------------------------------------------

_COMPARE_UNIT_DIMENSIONS = {
    "rent": "monthly_rent",
    "monthly_rent": "monthly_rent",
    "market_rent": "market_rent",
    "sqft": "sqft",
    "balance": "balance",
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
}


def compare_units(
    property_code: str,
    unit_numbers: list[str],
    dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """Side-by-side comparison of 2+ units within a property.

    Returns rent, sqft, market_rent, unit_type, lease_end and balance for
    each unit by default. `dimensions` filters which fields appear (the
    response composer uses this for the comparison chart).
    """
    code = require_scope(property_code)
    if not unit_numbers or len(unit_numbers) < 2:
        return {"property_code": code, "error": "compare_units requires at least 2 unit_numbers"}

    dims = [d.lower() for d in (dimensions or ["rent", "sqft", "market_rent"])]
    invalid = [d for d in dims if d not in _COMPARE_UNIT_DIMENSIONS]
    if invalid:
        return {
            "property_code": code,
            "error": f"Unknown dimension(s): {invalid}. Valid: {list(_COMPARE_UNIT_DIMENSIONS)}",
        }

    with session_scope() as s:
        # Pull units + their latest lease metadata in one go.
        rows = _rows_to_dicts(s.execute(
            text(f"""
                SELECT u.unit_number, u.unit_type, u.sqft, u.bedrooms, u.bathrooms,
                       u.market_rent,
                       l.monthly_rent, l.balance, l.lease_end, l.tenant_id, l.status
                FROM units u
                LEFT JOIN leases l
                  ON l.property_code = u.property_code AND l.unit_number = u.unit_number
                WHERE u.property_code = :code
                  AND u.unit_number IN :units
            """).bindparams(
                __import__("sqlalchemy").bindparam("units", expanding=True)
            ),
            {"code": code, "units": list(unit_numbers)},
        ))

    found_units = {r["unit_number"] for r in rows}
    missing = [u for u in unit_numbers if u not in found_units]

    # Reshape for charting: list of {dimension: {unit_number: value}}.
    series = []
    for dim in dims:
        col = _COMPARE_UNIT_DIMENSIONS[dim]
        series.append({
            "dimension": dim,
            "values": {r["unit_number"]: r.get(col) for r in rows},
        })

    return {
        "property_code": code,
        "unit_numbers": unit_numbers,
        "dimensions": dims,
        "missing": missing,
        "rows": [_stringify_dates(r) for r in rows],
        "series": series,
    }


# ---------------------------------------------------------------------------
# 9. Compare properties (cross-property aggregate)
# ---------------------------------------------------------------------------

_COMPARE_PROPERTY_DIMS = {
    "avg_rent": "AVG(CASE WHEN monthly_rent > 0 THEN monthly_rent END)",
    "total_units": "COUNT(*)",
    "occupied_units": "SUM(CASE WHEN occupied THEN 1 ELSE 0 END)",
    "occupancy_pct": (
        "ROUND(100.0 * SUM(CASE WHEN occupied THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1)"
    ),
    "rent_roll_total": "SUM(CASE WHEN monthly_rent > 0 THEN monthly_rent END)",
}


def compare_properties(
    property_codes: list[str],
    dimension: str = "avg_rent",
    month: str | None = None,
) -> dict[str, Any]:
    """Aggregate one dimension across multiple properties.

    Scope is the *list* of codes — every one is validated by require_scope.
    Defaults to the latest month per property.
    """
    if not property_codes or len(property_codes) < 2:
        return {"error": "compare_properties requires at least 2 property_codes"}
    codes = [require_scope(c) for c in property_codes]
    dim = (dimension or "avg_rent").lower()
    if dim not in _COMPARE_PROPERTY_DIMS:
        return {
            "error": f"Unknown dimension: {dim}. Valid: {list(_COMPARE_PROPERTY_DIMS)}",
        }
    agg_expr = _COMPARE_PROPERTY_DIMS[dim]

    # Resolve target month per code (latest if unspecified).
    with session_scope() as s:
        results = []
        for code in codes:
            if month:
                try:
                    y, m = month.split("-")[:2]
                    month_date = date(int(y), int(m), 1)
                except (ValueError, AttributeError):
                    return {"error": f"Invalid month: {month!r}"}
            else:
                month_date = s.execute(
                    text("SELECT MAX(snapshot_month) FROM rent_snapshots WHERE property_code = :c"),
                    {"c": code},
                ).scalar()

            if month_date is None:
                results.append({"property_code": code, "month": None, "value": None, "note": "no snapshots"})
                continue

            v = s.execute(
                text(f"""
                    SELECT {agg_expr} AS value
                    FROM rent_snapshots
                    WHERE property_code = :c AND snapshot_month = :m
                """),
                {"c": code, "m": month_date},
            ).scalar()
            results.append({
                "property_code": code,
                "month": month_date.isoformat(),
                "value": float(v) if v is not None else None,
            })

    return {
        "property_codes": codes,
        "dimension": dim,
        "month": month,
        "results": results,
    }


# ---------------------------------------------------------------------------
# 10. List units with flexible filters
# ---------------------------------------------------------------------------

def list_units(
    property_code: str,
    unit_type: str | None = None,
    bedrooms: float | None = None,
    min_rent: float | None = None,
    max_rent: float | None = None,
    occupied: bool | None = None,
    lease_ends_before: str | None = None,
    lease_ends_after: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Flexible filtered list of units for an analyst's exploratory queries."""
    code = require_scope(property_code)
    limit = max(1, min(int(limit or 50), 500))

    where = ["u.property_code = :code"]
    params: dict[str, Any] = {"code": code, "limit": limit}

    if unit_type:
        where.append("u.unit_type = :unit_type")
        params["unit_type"] = unit_type
    if bedrooms is not None:
        where.append("u.bedrooms = :bedrooms")
        params["bedrooms"] = float(bedrooms)
    if min_rent is not None:
        where.append("COALESCE(l.monthly_rent, u.market_rent) >= :min_rent")
        params["min_rent"] = float(min_rent)
    if max_rent is not None:
        where.append("COALESCE(l.monthly_rent, u.market_rent) <= :max_rent")
        params["max_rent"] = float(max_rent)
    if occupied is not None:
        where.append("(l.unit_number IS NOT NULL) = :occupied")
        params["occupied"] = bool(occupied)
    if lease_ends_before:
        where.append("l.lease_end <= :lease_ends_before")
        params["lease_ends_before"] = lease_ends_before
    if lease_ends_after:
        where.append("l.lease_end >= :lease_ends_after")
        params["lease_ends_after"] = lease_ends_after

    sql = f"""
        SELECT u.unit_number, u.unit_type, u.bedrooms, u.bathrooms, u.sqft,
               u.market_rent, l.monthly_rent, l.balance, l.lease_end,
               (l.unit_number IS NOT NULL) AS occupied
        FROM units u
        LEFT JOIN leases l
          ON l.property_code = u.property_code AND l.unit_number = u.unit_number
        WHERE {' AND '.join(where)}
        ORDER BY u.unit_number
        LIMIT :limit
    """
    with session_scope() as s:
        rows = _rows_to_dicts(s.execute(text(sql), params))
    rows = [_stringify_dates(r) for r in rows]
    return {
        "property_code": code,
        "filters": {k: v for k, v in params.items() if k not in ("code", "limit")},
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
    "get_unit_charges": {
        "fn": get_unit_charges,
        "description": (
            "Every charge line item for one unit, in source order. Preserves "
            "multiplicity — e.g. surfaces 2 PARKING lines of $75 and $100 "
            "rather than a single $175 total. Use this for 'what fees does "
            "unit X pay?' questions."
        ),
        "params": ["unit_number", "snapshot_month?"],
    },
    "compare_units": {
        "fn": compare_units,
        "description": (
            "Side-by-side comparison of 2+ units within ONE property. "
            "dimensions can include rent/sqft/market_rent/balance/bedrooms/bathrooms."
        ),
        "params": ["unit_numbers (list)", "dimensions (list, optional)"],
    },
    "compare_properties": {
        "fn": compare_properties,
        "description": (
            "Aggregate one metric across 2+ properties. dimension is one of "
            "avg_rent/total_units/occupied_units/occupancy_pct/rent_roll_total."
        ),
        "params": ["property_codes (list)", "dimension", "month?"],
    },
    "list_units": {
        "fn": list_units,
        "description": (
            "Filtered list of units. Filters: unit_type, bedrooms, min_rent, "
            "max_rent, occupied, lease_ends_before, lease_ends_after."
        ),
        "params": [
            "unit_type?", "bedrooms?", "min_rent?", "max_rent?",
            "occupied?", "lease_ends_before?", "lease_ends_after?", "limit?",
        ],
    },
    "execute_scoped_sql": {
        # Lazy import keeps the validator + reader connection out of startup.
        "fn": lambda *a, **kw: _lazy_execute_scoped_sql(*a, **kw),
        "description": (
            "BACKSTOP. Run a custom SELECT against the rent-roll DB when no "
            "curated tool fits. Tables: properties, units, leases, rent_snapshots, "
            "rent_charge_lines. The query is validated (sqlglot AST), scope-filtered, "
            "and executed as a read-only user. Use for novel multi-condition "
            "questions (e.g. 'units with rent > $2500 AND lease ending Q1 2026')."
        ),
        "params": ["sql"],  # property_codes injected by the graph
    },
}


def _lazy_execute_scoped_sql(property_codes, sql):
    from .sql_executor import execute_scoped_sql
    return execute_scoped_sql(property_codes, sql)


# NOTE: A legacy `run_tool()` string-keyed dispatcher used to live here.
# It was v1's hand-rolled router back when the LLM emitted JSON like
# `{"tool": "name", "args": {...}}`. v2 switched to native OpenAI
# tool-calling via `ChatOpenAI.bind_tools()`, which dispatches directly
# inside the `tools` graph node — `run_tool()` had zero callers and was
# removed in v3. The `TOOLS` registry above is still used by graph/nodes.py
# (`SQL_TOOLS` import) for `is this tool name a SQL tool?` lookups in
# `_route_label`, so the dict stays.
