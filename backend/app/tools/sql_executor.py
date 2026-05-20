"""LLM-written SQL backstop.

`execute_scoped_sql` is the agent's escape hatch when no curated tool fits.
It runs against a read-only MySQL user and goes through the sqlglot
validator first. Two defenses at two layers:

  Layer 1 (sqlglot):    AST checks — SELECT only, table allowlist,
                        scope filter required, no scope widening.
  Layer 2 (MySQL):      `property_reader` user has SELECT-only on the 5
                        whitelisted tables. Even if Layer 1 is bypassed
                        the DB rejects writes.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ..config import get_settings
from ..guardrails.scope import require_scope
from ..guardrails.sql_validator import (
    SqlValidationError,
    validate_and_rewrite,
)


@lru_cache(maxsize=1)
def _reader_session_factory():
    """Lazy-create a session bound to the read-only user."""
    settings = get_settings()
    engine = create_engine(
        settings.sqlalchemy_reader_url,
        pool_pre_ping=True,
        future=True,
    )
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


# Row cap returned to the caller even if validator allowed more.
MAX_RETURN_ROWS = 200


def execute_scoped_sql(
    property_codes: list[str] | str,
    sql: str,
) -> dict[str, Any]:
    """Run a validated SELECT scoped to one-or-many property codes.

    `property_codes` is a list — `compare_properties` style. A single string
    is accepted for ergonomic single-scope use.
    """
    if isinstance(property_codes, str):
        property_codes = [property_codes]
    # Validate each code is non-empty + normalized.
    codes = [require_scope(c) for c in (property_codes or [])]
    if not codes:
        return {"error": "execute_scoped_sql requires at least one property_code."}

    try:
        validated = validate_and_rewrite(sql, codes)
    except SqlValidationError as e:
        return {
            "error": f"SQL validation failed: {e}",
            "original_sql": sql,
            "scope_codes": codes,
        }

    Session = _reader_session_factory()
    truncated = False
    with Session() as s:
        try:
            result = s.execute(text(validated.sql))
            columns = list(result.keys())
            rows_raw = result.fetchmany(MAX_RETURN_ROWS + 1)
            if len(rows_raw) > MAX_RETURN_ROWS:
                rows_raw = rows_raw[:MAX_RETURN_ROWS]
                truncated = True
        except Exception as e:
            return {
                "error": f"Database error: {e}",
                "rewritten_sql": validated.sql,
                "scope_codes": codes,
            }

    rows = []
    for r in rows_raw:
        row_dict = {}
        for k, v in zip(columns, r):
            # Stringify dates/datetimes for JSON safety.
            if hasattr(v, "isoformat"):
                row_dict[k] = v.isoformat()
            else:
                row_dict[k] = v
        rows.append(row_dict)

    return {
        "scope_codes": codes,
        "rewritten_sql": validated.sql,
        "auto_injected_filters": bool(validated.auto_injected_filters),
        "auto_injected_limit": validated.auto_injected_limit,
        "referenced_tables": validated.referenced_tables,
        "columns": columns,
        "row_count": len(rows),
        "truncated": truncated,
        "rows": rows,
    }
