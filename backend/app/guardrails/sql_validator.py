"""SQL validator + scope-injector for LLM-written queries.

This file is the only thing between the LLM's free-form SQL and the database.
It enforces, in order:

  1. Single-statement SELECT only (no DDL/DML/multi-statement).
  2. Table allowlist (`properties, units, leases, rent_snapshots,
     rent_charge_lines`). Any other identifier in a FROM/JOIN -> reject.
  3. Every reference to a scoped table requires a
     `property_code IN (<allowed>)` filter. Auto-injected when missing.
  4. No scope-widening predicates: a literal `property_code` comparison can
     ONLY reference codes in the caller's allow-list.
  5. LIMIT added if absent (cap 500 rows).

Defense-in-depth: even if this layer is fooled, the database connection runs
as the `property_reader` Postgres role which has SELECT-only on these five
tables.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


# Tables the LLM is allowed to reference.
ALLOWED_TABLES: frozenset[str] = frozenset({
    "properties",
    "units",
    "leases",
    "rent_snapshots",
    "rent_charge_lines",
})

# Tables that carry a `property_code` column. Every reference must be scope-filtered.
SCOPED_TABLES: frozenset[str] = ALLOWED_TABLES  # all five carry property_code

DEFAULT_LIMIT = 500


class SqlValidationError(ValueError):
    """Raised when an LLM-written SQL fails the safety pipeline."""


@dataclass
class ValidatedSql:
    sql: str                       # the rewritten, safe SQL ready to run
    referenced_tables: list[str]
    scope_codes: list[str]
    auto_injected_filters: list[str]
    auto_injected_limit: bool


def _is_select_only(tree: exp.Expression) -> None:
    """Reject anything that's not a single read."""
    # The top-level node should be Select, Union, Subquery, or CTE-wrapped Select.
    allowed_top = (exp.Select, exp.Union, exp.With, exp.Subquery)
    if not isinstance(tree, allowed_top):
        raise SqlValidationError(
            f"Only SELECT statements are allowed (got {type(tree).__name__})."
        )
    # Walk the AST: any DML/DDL node anywhere = reject.
    forbidden = (
        exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
        exp.Create, exp.TruncateTable, exp.Merge,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise SqlValidationError(
                f"Statement type '{type(node).__name__}' is not allowed."
            )


def _collect_tables(tree: exp.Expression) -> list[exp.Table]:
    return [t for t in tree.find_all(exp.Table)]


def _check_table_allowlist(tables: list[exp.Table]) -> list[str]:
    names: list[str] = []
    for t in tables:
        name = t.name.lower()
        if name not in ALLOWED_TABLES:
            raise SqlValidationError(
                f"Table {name!r} is not allowed. Allowed: {sorted(ALLOWED_TABLES)}."
            )
        names.append(name)
    return names


def _literal_codes_in_predicate(node: exp.Expression) -> set[str]:
    """Extract every string literal compared against a property_code column."""
    found: set[str] = set()
    for cmp in node.find_all(exp.EQ, exp.In, exp.NEQ):
        left = cmp.args.get("this")
        right = cmp.args.get("expression") or cmp.args.get("expressions")

        def col_is_property_code(e: exp.Expression | None) -> bool:
            return isinstance(e, exp.Column) and e.name.lower() == "property_code"

        # Equality / inequality: `property_code = '115r'` or '115r' = property_code
        if col_is_property_code(left) and isinstance(right, exp.Literal):
            found.add(right.this.lower())
        elif col_is_property_code(right) and isinstance(left, exp.Literal):
            found.add(left.this.lower())
        # IN clause: `property_code IN ('115r','126r')`
        elif isinstance(cmp, exp.In) and col_is_property_code(left):
            for lit in cmp.args.get("expressions") or []:
                if isinstance(lit, exp.Literal):
                    found.add(lit.this.lower())
    return found


def _check_no_scope_widening(tree: exp.Expression, allowed_codes: set[str]) -> None:
    """Every literal compared to property_code must be in the allow-list."""
    literals = _literal_codes_in_predicate(tree)
    extras = literals - allowed_codes
    if extras:
        raise SqlValidationError(
            f"Query references property_code(s) outside the active scope: "
            f"{sorted(extras)}. Allowed: {sorted(allowed_codes)}."
        )


def _ensure_scope_filter(
    tree: exp.Expression, allowed_codes: set[str]
) -> list[str]:
    """For every SELECT, AND in a property_code IN (...) filter if absent.

    This is the heart of the scope guarantee: even if the LLM forgets to add
    the WHERE clause, we add it.

    Returns the list of selects we mutated (for reporting).
    """
    injected: list[str] = []
    codes_list = sorted(allowed_codes)

    for select in tree.find_all(exp.Select):
        # Only inject when this SELECT actually references a table. sqlglot's
        # arg key for the FROM clause shifted between versions (`from` vs
        # `from_`), so we check for Table descendants directly instead.
        if not list(select.find_all(exp.Table)):
            continue
        # Build the IN-list expression once.
        in_expr = exp.In(
            this=exp.Column(this=exp.to_identifier("property_code")),
            expressions=[exp.Literal.string(c) for c in codes_list],
        )

        # If this select already has any property_code predicate, leave it
        # (we already validated no widening). Otherwise inject.
        where = select.args.get("where")
        existing_codes = _literal_codes_in_predicate(where) if where else set()
        if existing_codes:
            continue

        if where is None:
            select.set("where", exp.Where(this=in_expr))
        else:
            select.set("where", exp.Where(this=exp.And(this=where.this, expression=in_expr)))
        injected.append(select.sql(dialect="postgres"))
    return injected


def _ensure_limit(tree: exp.Expression) -> bool:
    """Add LIMIT 500 if there's no LIMIT at the outermost SELECT."""
    if not isinstance(tree, (exp.Select, exp.Union)):
        return False
    if tree.args.get("limit") is not None:
        return False
    tree.set("limit", exp.Limit(expression=exp.Literal.number(DEFAULT_LIMIT)))
    return True


def validate_and_rewrite(
    sql: str,
    allowed_property_codes: list[str],
) -> ValidatedSql:
    """Run the full pipeline. Raises SqlValidationError on any failure."""
    if not sql or not sql.strip():
        raise SqlValidationError("SQL is empty.")
    if not allowed_property_codes:
        raise SqlValidationError("At least one allowed property_code is required.")

    allowed_codes = {c.strip().lower() for c in allowed_property_codes}

    # Parse with the Postgres dialect.
    try:
        trees = sqlglot.parse(sql, dialect="postgres")
    except Exception as e:
        raise SqlValidationError(f"Failed to parse SQL: {e}") from e
    if len(trees) != 1:
        raise SqlValidationError(
            f"Multi-statement SQL not allowed (found {len(trees)} statements)."
        )
    tree = trees[0]
    if tree is None:
        raise SqlValidationError("Empty parse result.")

    _is_select_only(tree)
    tables = _collect_tables(tree)
    if not tables:
        raise SqlValidationError("No table references found in query.")
    table_names = _check_table_allowlist(tables)
    _check_no_scope_widening(tree, allowed_codes)
    injected = _ensure_scope_filter(tree, allowed_codes)
    limit_added = _ensure_limit(tree)

    return ValidatedSql(
        sql=tree.sql(dialect="postgres"),
        referenced_tables=table_names,
        scope_codes=sorted(allowed_codes),
        auto_injected_filters=injected,
        auto_injected_limit=limit_added,
    )
