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

import re
from dataclasses import dataclass, field
from typing import Literal

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


# ---------------------------------------------------------------------------
# v2: scope extraction from the user's natural-language message
# ---------------------------------------------------------------------------

# Property codes follow two patterns:
#   - 3 digits + r/a/c (residential / affordable / commercial), e.g. "115r", "126a"
#   - 3 digits + "land", e.g. "134land"
#   - literal "altapm"
#
# Word boundaries on both sides keep us from matching things like "1234r" or
# "abc115ra". We do NOT require leading/trailing whitespace because users
# write things like "115r's amenities" or "(115r)".
_CODE_PATTERN = re.compile(
    r"\b(?:\d{3}(?:r|a|c|land)|altapm)\b",
    re.IGNORECASE,
)


def extract_codes_from_message(message: str) -> list[str]:
    """Pull all property codes from a user message. Validates against the DB.

    Returns lowercased, deduplicated, only codes that exist in `properties`.
    Order preserved by first occurrence. Empty list if nothing valid found.
    """
    if not message:
        return []
    candidates: list[str] = []
    seen: set[str] = set()
    for m in _CODE_PATTERN.finditer(message):
        code = m.group(0).lower()
        if code not in seen:
            seen.add(code)
            candidates.append(code)
    if not candidates:
        return []
    # Cross-check against the catalogue — anything not in the DB is dropped.
    with session_scope() as s:
        valid = {
            r[0] for r in s.execute(
                select(Property.property_code).where(Property.property_code.in_(candidates))
            ).all()
        }
    return [c for c in candidates if c in valid]


# Scope-decision union (frozen — passed across graph nodes)

ScopeKind = Literal["single", "compare", "conflict", "missing"]


@dataclass
class ScopeDecision:
    """What the scope-resolver decided about the active query.

    - single   : exactly one code identified. `code` is set.
    - compare  : 2+ codes in the message → the agent will compare them.
                 `codes` is set.
    - conflict : dropdown says X but message mentions Y. `dropdown_code` and
                 `query_code` are both set; the graph must interrupt and ask
                 the user which one to use.
    - missing  : no code anywhere. `available` lists all property codes so the
                 frontend can render a picker; graph must interrupt.
    """
    kind: ScopeKind
    code: str | None = None
    codes: list[str] = field(default_factory=list)
    dropdown_code: str | None = None
    query_code: str | None = None
    available: list[str] = field(default_factory=list)
    source: Literal["query", "dropdown", "resumed", None] = None

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "code": self.code,
            "codes": self.codes,
            "dropdown_code": self.dropdown_code,
            "query_code": self.query_code,
            "available": self.available,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScopeDecision":
        return cls(
            kind=d.get("kind", "missing"),
            code=d.get("code"),
            codes=d.get("codes") or [],
            dropdown_code=d.get("dropdown_code"),
            query_code=d.get("query_code"),
            available=d.get("available") or [],
            source=d.get("source"),
        )


def _all_property_codes() -> list[str]:
    with session_scope() as s:
        return [
            r[0] for r in s.execute(
                select(Property.property_code).order_by(Property.property_code)
            ).all()
        ]


def resolve_scope(
    dropdown_code: str | None,
    message: str,
) -> ScopeDecision:
    """Combine dropdown + message into one ScopeDecision.

    Resolution rules (matches the v2 plan):

      A. Multiple valid codes in message  -> `compare`
      B. One valid code in message
            - matches dropdown            -> `single`(source="query")
            - dropdown empty              -> `single`(source="query")
            - mismatches dropdown         -> `conflict`  (ask user)
      C. No valid code in message
            - dropdown set and valid      -> `single`(source="dropdown")
            - dropdown empty              -> `missing`  (ask user)
    """
    msg_codes = extract_codes_from_message(message)
    dropdown_clean = (dropdown_code or "").strip().lower() or None
    if dropdown_clean:
        # If the dropdown value isn't a real code, treat as if empty.
        try:
            validate_property_code(dropdown_clean)
        except (UnknownPropertyError, ScopeViolationError):
            dropdown_clean = None

    # Case A: multiple codes in message
    if len(msg_codes) > 1:
        return ScopeDecision(kind="compare", codes=msg_codes, source="query")

    # Case B: exactly one code in message
    if len(msg_codes) == 1:
        q = msg_codes[0]
        if dropdown_clean and dropdown_clean != q:
            return ScopeDecision(
                kind="conflict",
                dropdown_code=dropdown_clean,
                query_code=q,
            )
        return ScopeDecision(kind="single", code=q, source="query")

    # Case C: no codes in message
    if dropdown_clean:
        return ScopeDecision(kind="single", code=dropdown_clean, source="dropdown")
    return ScopeDecision(
        kind="missing",
        available=_all_property_codes(),
    )
