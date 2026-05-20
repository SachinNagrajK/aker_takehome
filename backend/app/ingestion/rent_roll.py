"""Rent-roll ingestion.

Parses the 300 monthly rent-roll spreadsheets shipped with the assignment and
loads them into MySQL.

Per-file layout (consistent across all 25 properties × 12 months):

  Row 0:  "Rent Roll with Lease Charges"
  Row 1:  "<Property Name> (<code>)"            e.g. "Canfield Park (115r)"
  Row 2:  "As Of = MM/DD/YYYY"
  Row 3:  "Month Year = MM/YYYY"                 <-- snapshot month
  Row 4-5: two-row header
  Row 6+: data rows in section "Current/Notice/Vacant Residents"
          then "Future Residents/Applicants"
          then "Totals:" / "Summary Groups" (ignored)

Each *unit* is a multi-row block: the first row has the unit identity columns
populated plus the first charge code (often RENT). Subsequent rows have only
the ChargeCode + Amount populated (PARKING, TRASH, AMENITY, PETFEEM, ...).
The block ends with a `Total` row.

Run:
    python -m app.ingestion.rent_roll
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.dialects.mysql import insert as mysql_insert

# Allow `python -m app.ingestion.rent_roll` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import get_settings        # noqa: E402
from app.db import engine, session_scope   # noqa: E402
from app import models                      # noqa: E402


# ---------------------------------------------------------------------------
# Filename / metadata parsing
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^(?P<month>[A-Z][a-z]{2})_RENT_ROLL_WITH_LEASE_CHARGES_(?P<code>[A-Za-z0-9]+)\.xls$"
)

MONTH_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

TITLE_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<code>[A-Za-z0-9]+)\)\s*$")
MONTH_YEAR_RE = re.compile(r"Month\s*Year\s*=\s*(?P<m>\d{1,2})/(?P<y>\d{4})")


def _classify_property_type(code: str) -> str:
    """Infer property type from the code suffix."""
    if code.endswith("land"):
        return "land"
    if code and code[-1] in {"r", "a", "c"}:
        return {"r": "residential", "a": "affordable", "c": "commercial"}[code[-1]]
    return "other"


# ---------------------------------------------------------------------------
# Per-file parser
# ---------------------------------------------------------------------------

# Logical column index (0-based, matching the source sheet's 14 columns).
COL_UNIT          = 0
COL_UNIT_TYPE     = 1
COL_SQFT          = 2
COL_RESIDENT      = 3
COL_NAME          = 4
COL_MARKET_RENT   = 5
COL_CHARGE_CODE   = 6
COL_AMOUNT        = 7
COL_RESIDENT_DEP  = 8
COL_OTHER_DEP     = 9
COL_MOVE_IN       = 10
COL_LEASE_EXP     = 11
COL_MOVE_OUT      = 12
COL_BALANCE       = 13

SECTION_CURRENT = "Current/Notice/Vacant Residents"
SECTION_FUTURE  = "Future Residents/Applicants"

# Hard-stop markers: when col 0 matches any of these, abandon parsing
# immediately regardless of whether we entered the data section. This catches
# summary-only files (e.g. 134land, altapm) where "Current/Notice/Vacant
# Residents" appears as a row *inside* the Summary Groups table at the bottom,
# which would otherwise re-trigger data parsing.
HARD_STOP_PREFIXES = ("Summary Groups", "Totals:", "Summary of Charges")

# "Rent" can be billed under different charge codes depending on property type:
#   RENT      - residential
#   RENTAFF   - affordable housing
#   RENTRETL  - retail / commercial
# Sum across these per unit block. Concessions live in CONCESSION_CODES.
RENT_CHARGE_CODES = {"RENT", "RENTAFF", "RENTRETL"}
CONCESSION_CODES  = {"CONRENT", "CONRETL"}


@dataclass
class ParsedFile:
    property_code: str
    property_name: str
    snapshot_month: date
    units: list[dict] = field(default_factory=list)       # one per unit block
    rows: list[dict] = field(default_factory=list)        # one snapshot row per unit


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    if not s or s.lower() in {"nan", "vacant"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_date(v) -> date | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return pd.to_datetime(v, errors="coerce").date()
    except Exception:
        return None


def _str(v) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def parse_rent_roll_file(path: Path) -> ParsedFile:
    df = pd.read_excel(path, header=None, engine="openpyxl")

    # Title -> property name & code (more reliable than filename for the name).
    title = str(df.iat[1, 0]) if df.shape[0] > 1 else ""
    m = TITLE_RE.match(title)
    if m:
        property_name = m.group("name").strip()
        code_from_title = m.group("code").strip().lower()
    else:
        property_name = title.strip() or "Unknown"
        code_from_title = ""

    # Snapshot month from row 3.
    snap_row = str(df.iat[3, 0]) if df.shape[0] > 3 else ""
    mm = MONTH_YEAR_RE.search(snap_row)
    if not mm:
        raise ValueError(f"{path.name}: could not parse Month Year row: {snap_row!r}")
    snapshot_month = date(int(mm.group("y")), int(mm.group("m")), 1)

    # Code from filename as authoritative (handles edge cases like "altapm").
    fn_match = FILENAME_RE.match(path.name)
    code = fn_match.group("code").lower() if fn_match else code_from_title
    if not code:
        raise ValueError(f"{path.name}: could not determine property_code")

    result = ParsedFile(
        property_code=code,
        property_name=property_name,
        snapshot_month=snapshot_month,
    )

    # Walk data rows. Each unit "block" starts on a row with a non-empty Unit
    # column and continues until the next non-empty Unit row, a section marker,
    # or a stop marker.
    in_data = False
    current: dict | None = None

    def _record_charge(block: dict, code_upper: str, amount: float) -> None:
        # Append a granular line item AND update the summary dict.
        # `charge_lines` preserves multiplicity (e.g. two PARKING rows
        # appearing separately); the dict gives an O(1) per-code total.
        line_index = len(block["charge_lines"])
        block["charge_lines"].append({
            "line_index": line_index,
            "charge_code": code_upper,
            "amount": amount,
        })
        block["charges"][code_upper] = block["charges"].get(code_upper, 0.0) + amount
        if code_upper in RENT_CHARGE_CODES:
            block["monthly_rent"] = (block["monthly_rent"] or 0.0) + amount

    for i in range(4, len(df)):  # data starts after header rows
        row = df.iloc[i]
        c0 = _str(row.iat[COL_UNIT])

        # Hard stop: any summary/totals marker ends parsing for this file.
        # Must be checked *before* the SECTION_CURRENT check, because some
        # files contain a second "Current/Notice/Vacant Residents" row inside
        # the Summary Groups table that would otherwise re-trigger data mode.
        if c0 and any(c0.startswith(s) for s in HARD_STOP_PREFIXES):
            current = None
            break

        # Section transitions
        if c0 == SECTION_CURRENT:
            in_data = True
            current = None
            continue
        if c0 == SECTION_FUTURE:
            # Future tenants are out-of-scope for occupancy/current-rent stats.
            current = None
            break
        if not in_data:
            continue

        # Wholly blank line
        if all(_str(row.iat[k]) is None for k in range(min(8, df.shape[1]))):
            continue

        if c0:
            # New unit block.
            unit_number = c0
            unit_type = _str(row.iat[COL_UNIT_TYPE])
            sqft = _to_float(row.iat[COL_SQFT])
            resident = _str(row.iat[COL_RESIDENT])
            market_rent = _to_float(row.iat[COL_MARKET_RENT])

            is_vacant = (resident or "").upper() == "VACANT"
            current = {
                "unit_number": unit_number,
                "unit_type": unit_type,
                "sqft": sqft,
                "market_rent": market_rent,
                "tenant_id": None if is_vacant else resident,
                "occupied": not is_vacant,
                "move_in": _to_date(row.iat[COL_MOVE_IN]),
                "lease_end": _to_date(row.iat[COL_LEASE_EXP]),
                "move_out": _to_date(row.iat[COL_MOVE_OUT]),
                "balance": _to_float(row.iat[COL_BALANCE]),
                "monthly_rent": None,
                "charges": {},        # per-code totals (summary)
                "charge_lines": [],   # ordered per-line items (granular truth)
            }
            cc = _str(row.iat[COL_CHARGE_CODE])
            amt = _to_float(row.iat[COL_AMOUNT])
            if cc and amt is not None:
                _record_charge(current, cc.upper(), amt)
            result.units.append(current)
        else:
            # Continuation row inside current block (PARKING, AMENITY, etc.).
            if current is None:
                continue
            cc = _str(row.iat[COL_CHARGE_CODE])
            amt = _to_float(row.iat[COL_AMOUNT])
            if cc is None:
                continue
            cc_u = cc.upper()
            if cc_u == "TOTAL":
                current = None  # block ends; next unit row starts a new block
                continue
            if amt is not None:
                _record_charge(current, cc_u, amt)

    # Build snapshot rows (one per unit). `charge_lines` rides along so the
    # writer can persist them in the new rent_charge_lines table.
    for u in result.units:
        charges = u["charges"]
        concessions = sum(v for k, v in charges.items() if k in CONCESSION_CODES)
        effective_rent = (
            (u["monthly_rent"] + concessions) if u["monthly_rent"] is not None else None
        )
        result.rows.append({
            "unit_number": u["unit_number"],
            "monthly_rent": u["monthly_rent"],
            "occupied": u["occupied"],
            "charge_lines": u["charge_lines"],  # NEW: per-line items in source order
            "raw_row": {
                "charges": charges,
                "market_rent": u["market_rent"],
                "tenant_id": u["tenant_id"],
                "balance": u["balance"],
                "effective_rent": effective_rent,   # rent net of concessions
                "concessions": concessions if concessions else None,
                "move_in": u["move_in"].isoformat() if u["move_in"] else None,
                "lease_end": u["lease_end"].isoformat() if u["lease_end"] else None,
                "move_out": u["move_out"].isoformat() if u["move_out"] else None,
            },
        })

    return result


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _upsert_property(session, code: str, name: str) -> None:
    stmt = mysql_insert(models.Property).values(
        property_code=code,
        property_name=name,
        property_type=_classify_property_type(code),
    )
    stmt = stmt.on_duplicate_key_update(
        property_name=stmt.inserted.property_name,
        property_type=stmt.inserted.property_type,
    )
    session.execute(stmt)


def _replace_snapshot(session, code: str, month: date, rows: list[dict]) -> int:
    """Delete + insert all rent_snapshot rows for (code, month). Idempotent.

    Also rewrites rent_charge_lines for this (code, month). Returns the
    number of charge lines written so the caller can report a summary.

    ON DELETE CASCADE on rent_charge_lines.snapshot_id means the explicit
    rent_snapshot delete also clears its lines — but we issue an explicit
    delete on rent_charge_lines too in case any orphans exist from a
    previous bulk-insert path.
    """
    # 1) Wipe existing rows for this (code, month).
    session.execute(
        delete(models.RentChargeLine).where(
            models.RentChargeLine.property_code == code,
            models.RentChargeLine.snapshot_month == month,
        )
    )
    session.execute(
        delete(models.RentSnapshot).where(
            models.RentSnapshot.property_code == code,
            models.RentSnapshot.snapshot_month == month,
        )
    )
    if not rows:
        return 0

    # 2) Insert snapshots one-by-one so we can capture each PK to associate
    #    its charge lines. bulk_insert_mappings doesn't return PKs reliably
    #    across DBs, and we need them for the FK.
    line_payload: list[dict] = []
    for r in rows:
        snap = models.RentSnapshot(
            property_code=code,
            snapshot_month=month,
            unit_number=r["unit_number"],
            monthly_rent=r["monthly_rent"],
            occupied=r["occupied"],
            raw_row=r["raw_row"],
        )
        session.add(snap)
        session.flush()  # populates snap.id
        for cl in r.get("charge_lines", []):
            line_payload.append({
                "snapshot_id": snap.id,
                "property_code": code,
                "snapshot_month": month,
                "unit_number": r["unit_number"],
                "line_index": cl["line_index"],
                "charge_code": cl["charge_code"],
                "amount": cl["amount"],
            })

    if line_payload:
        session.bulk_insert_mappings(models.RentChargeLine, line_payload)

    return len(line_payload)


def _refresh_units_and_leases(session, code: str, latest_units: list[dict]) -> None:
    """Replace `units` and `leases` with the most-recent snapshot's data."""
    # Wipe existing per-property rows.
    session.execute(delete(models.Unit).where(models.Unit.property_code == code))
    session.execute(delete(models.Lease).where(models.Lease.property_code == code))

    unit_rows = []
    lease_rows = []
    for u in latest_units:
        unit_rows.append({
            "property_code": code,
            "unit_number": u["unit_number"],
            "unit_type": u["unit_type"],
            "bedrooms": None,
            "bathrooms": None,
            "sqft": u["sqft"],
            "market_rent": u["market_rent"],
        })
        if u["occupied"]:
            lease_rows.append({
                "property_code": code,
                "unit_number": u["unit_number"],
                "tenant_id": u["tenant_id"],
                "lease_start": u["move_in"],
                "lease_end": u["lease_end"],
                "monthly_rent": u["monthly_rent"],
                "balance": u["balance"],
                "status": "current",
            })
    if unit_rows:
        session.bulk_insert_mappings(models.Unit, unit_rows)
    if lease_rows:
        session.bulk_insert_mappings(models.Lease, lease_rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def ingest_directory(rent_roll_dir: Path) -> None:
    files = sorted(p for p in rent_roll_dir.glob("*.xls"))
    if not files:
        raise SystemExit(f"No .xls files found in {rent_roll_dir}")

    print(f"Found {len(files)} files in {rent_roll_dir}")

    # Group by property code so we can pick the latest snapshot for each.
    parsed_by_code: dict[str, list[ParsedFile]] = {}

    from app.db import init_db
    init_db()

    n_props = n_snap_rows = n_line_rows = 0
    with session_scope() as session:
        for i, path in enumerate(files, 1):
            try:
                pf = parse_rent_roll_file(path)
            except Exception as e:
                print(f"  [!] {path.name}: {e}")
                continue

            _upsert_property(session, pf.property_code, pf.property_name)
            n_line_rows += _replace_snapshot(session, pf.property_code, pf.snapshot_month, pf.rows)
            parsed_by_code.setdefault(pf.property_code, []).append(pf)
            n_snap_rows += len(pf.rows)

            if i % 25 == 0 or i == len(files):
                print(f"  parsed {i}/{len(files)} files")

        # For each property, rebuild units/leases from the most recent snapshot.
        for code, parsed_list in parsed_by_code.items():
            latest = max(parsed_list, key=lambda p: p.snapshot_month)
            _refresh_units_and_leases(session, code, latest.units)
            n_props += 1

    print()
    print(f"Properties loaded:   {n_props}")
    print(f"Files ingested:      {sum(len(v) for v in parsed_by_code.values())}")
    print(f"Snapshot rows:       {n_snap_rows}")
    print(f"Charge-line rows:    {n_line_rows}")
    print(f"Property codes:      {', '.join(sorted(parsed_by_code))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest rent-roll spreadsheets into MySQL.")
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path(get_settings().rent_roll_dir),
        help="Directory containing the *.xls files (default from RENT_ROLL_DIR env).",
    )
    args = ap.parse_args()
    ingest_directory(args.dir.resolve())


if __name__ == "__main__":
    main()
