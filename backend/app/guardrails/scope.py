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


def _normalize_dropdown(dropdown_code) -> list[str]:
    """Accept either a single code or a list (multi-select). Returns valid lower-cased codes."""
    if not dropdown_code:
        return []
    raw = dropdown_code if isinstance(dropdown_code, list) else [dropdown_code]
    out: list[str] = []
    seen: set[str] = set()
    for v in raw:
        if not isinstance(v, str):
            continue
        c = v.strip().lower()
        if not c or c in seen:
            continue
        try:
            validate_property_code(c)
        except (UnknownPropertyError, ScopeViolationError):
            continue
        seen.add(c)
        out.append(c)
    return out


def resolve_scope(
    dropdown_code,
    message: str,
) -> ScopeDecision:
    """Combine dropdown + message into one ScopeDecision.

    Smart resolution rules:

      - Union of dropdown codes + message-mentioned codes (deduped, ordered)
      - 2+ codes -> `compare`
      - 1 code   -> `single`
      - 0 codes  -> `missing` (ask user)

    Dropdown + message disagreement is no longer a conflict — we now treat
    "I'm scoped to X but asking about Y" as a request to consider both.
    The user explicitly controls multi-scope via the multi-select dropdown.
    """
    msg_codes = extract_codes_from_message(message)
    dropdown_codes = _normalize_dropdown(dropdown_code)

    union: list[str] = []
    for c in dropdown_codes + msg_codes:
        if c not in union:
            union.append(c)

    if len(union) >= 2:
        source = "query" if msg_codes and not dropdown_codes else "dropdown"
        return ScopeDecision(kind="compare", codes=union, source=source)

    if len(union) == 1:
        code = union[0]
        if msg_codes and not dropdown_codes:
            return ScopeDecision(kind="single", code=code, source="query")
        return ScopeDecision(kind="single", code=code, source="dropdown")

    return ScopeDecision(
        kind="missing",
        available=_all_property_codes(),
    )
