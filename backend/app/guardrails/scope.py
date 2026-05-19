"""Property-scope guardrail.

This is the single chokepoint enforcing the assignment's hard requirement:

> Property scoping must be enforced end-to-end: every query, retrieval call,
> and tool invocation should be bounded to the active property code.

Every SQL tool, every RAG call, and every LLM prompt MUST route through one
of the helpers here. The contract:

- `validate_property_code(code)` raises `UnknownPropertyError` if the code
  isn't in the `properties` table. Called once at the start of every chat
  request.
- `require_scope(code)` is a runtime assertion every tool calls as its first
  line — if a tool ever ends up with an empty code, this fails loudly.
- `system_prompt(code, name)` produces the templated system prompt that pins
  the LLM to a single property.
"""
from __future__ import annotations

from sqlalchemy import select

from ..db import session_scope
from ..models import Property


class UnknownPropertyError(ValueError):
    """Raised when a property_code does not exist in the catalog."""


class ScopeViolationError(RuntimeError):
    """Raised when a tool is invoked without a property_code."""


def require_scope(property_code: str | None) -> str:
    """Assert that a non-empty property_code is present.

    Every SQL/RAG tool calls this as its first line. Centralising it means
    one place to read in code review when checking "is scoping enforced?".
    """
    if not property_code or not isinstance(property_code, str):
        raise ScopeViolationError(
            "property_code is required for every tool invocation"
        )
    code = property_code.strip().lower()
    if not code:
        raise ScopeViolationError("property_code is empty after normalisation")
    return code


def validate_property_code(property_code: str) -> tuple[str, str]:
    """Verify the code exists in MySQL. Returns (code, property_name).

    Called by the API layer before any tool runs. This is the only place we
    touch the DB for the existence check — failed checks short-circuit the
    whole pipeline with a 400-style error.
    """
    code = require_scope(property_code)
    with session_scope() as s:
        row = s.execute(
            select(Property.property_code, Property.property_name).where(
                Property.property_code == code
            )
        ).first()
    if row is None:
        raise UnknownPropertyError(f"Unknown property_code: {property_code!r}")
    return row[0], row[1]


SYSTEM_PROMPT_TEMPLATE = """\
You are a property-management AI assistant. Answer ONLY about the property
identified by code "{code}" — known as "{name}".

Hard rules:
  1. Never mention or reference any other property in the portfolio.
  2. If the data needed to answer is unavailable for THIS property, say so
     explicitly. Do not guess, infer, or borrow from training data.
  3. The tools you can call already enforce a property_code = "{code}" filter
     at the SQL and vector-store level. Trust their output.
  4. Format your answer in Markdown. When numeric data is involved, also
     surface UI components (KPI cards, tables, charts) so the user gets
     both a narrative and structured view.

The user is operating in a single-property session. Stay scoped."""


def system_prompt(property_code: str, property_name: str) -> str:
    code = require_scope(property_code)
    return SYSTEM_PROMPT_TEMPLATE.format(code=code, name=property_name)
