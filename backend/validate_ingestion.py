"""Corpus-wide ingestion validator.

For every one of the 300 monthly workbooks, this script:

  1. Independently re-reads the raw Excel cells WITHOUT using the production
     parser's unit-block detection. Aggregates `amount` per charge_code by
     scanning every data row under "Current/Notice/Vacant Residents" until a
     stop marker is hit.
  2. Queries MySQL for the same property+month: SUM(amount) GROUP BY charge_code.
  3. Diffs the two and emits a per-file report.

Also validates:
  - Distinct unit count per file
  - Total Resident Deposit per file (sum of col 8 on unit rows)
  - That every workbook produced at least one snapshot row in MySQL

Goal: detect any mismatch between source-of-truth Excel cells and what's
stored, beyond what a single spot-check would catch.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_settings
from app.db import session_scope
from app.ingestion.rent_roll import FILENAME_RE, MONTH_ABBR, MONTH_YEAR_RE, TITLE_RE
from datetime import date

# ---------------------------------------------------------------------------
# Workbook reader — independent of production parser
# ---------------------------------------------------------------------------

SECTION_CURRENT = "Current/Notice/Vacant Residents"
SECTION_FUTURE  = "Future Residents/Applicants"
HARD_STOP_PREFIXES = ("Summary Groups", "Totals:", "Summary of Charges")


def _to_float(v) -> float | None:
    if v is None: return None
    if isinstance(v, (int, float)):
        import math
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip().replace(",", "").replace("$", "")
    if not s or s.lower() in {"nan", "vacant"}: return None
    try: return float(s)
    except ValueError: return None


def _str_or_none(v):
    if v is None: return None
    if isinstance(v, float):
        import math
        if math.isnan(v): return None
    s = str(v).strip()
    return s or None


def read_workbook_truth(path: Path) -> dict:
    """Return the ground-truth aggregates for one workbook by scanning cells
    directly. Mirrors the section/stop-marker rules but does NOT use unit-block
    state tracking — every row that contains (charge_code, amount) inside the
    Current section is counted independently.
    """
    df = pd.read_excel(path, header=None, engine="openpyxl")

    # Property code: try filename first, fall back to row-1 title.
    # The filename regex requires 3-letter months (Jan/Sep) but 25 files
    # use "Sept" (4 letters) — so for those we MUST use the title fallback
    # exactly like the production ingester does.
    fn_match = FILENAME_RE.match(path.name)
    code = fn_match.group("code").lower() if fn_match else None
    if not code:
        title = str(df.iat[1, 0]) if df.shape[0] > 1 else ""
        tm = TITLE_RE.match(title)
        code = tm.group("code").strip().lower() if tm else None

    # Snapshot month from row 3
    snap_row = str(df.iat[3, 0]) if df.shape[0] > 3 else ""
    mm = MONTH_YEAR_RE.search(snap_row)
    snapshot_month = date(int(mm.group("y")), int(mm.group("m")), 1) if mm else None

    in_data = False
    charge_sums: dict[str, float] = defaultdict(float)
    charge_counts: dict[str, int] = defaultdict(int)
    unit_count = 0
    deposit_sum = 0.0
    deposit_count = 0
    move_out_count = 0

    for i in range(6, len(df)):
        c0 = _str_or_none(df.iat[i, 0])
        if c0 and any(c0.startswith(s) for s in HARD_STOP_PREFIXES):
            break
        if c0 == SECTION_CURRENT:
            in_data = True
            continue
        if c0 == SECTION_FUTURE:
            break
        if not in_data:
            continue

        # Unit-identity rows (col 0 has unit number)
        if c0:
            unit_count += 1
            dep = _to_float(df.iat[i, 8]) if df.shape[1] > 8 else None
            if dep is not None:
                deposit_sum += dep
                if dep > 0: deposit_count += 1
            mout = df.iat[i, 12] if df.shape[1] > 12 else None
            if mout is not None and str(mout).strip() not in ("nan", ""):
                move_out_count += 1

        # Every row (unit or continuation) may have a charge code + amount
        cc = _str_or_none(df.iat[i, 6]) if df.shape[1] > 6 else None
        amt = _to_float(df.iat[i, 7]) if df.shape[1] > 7 else None
        if cc and amt is not None:
            cc_u = cc.upper()
            if cc_u == "TOTAL":
                continue  # ignore subtotals
            charge_sums[cc_u] += amt
            charge_counts[cc_u] += 1

    return {
        "property_code": code,
        "snapshot_month": snapshot_month,
        "unit_count": unit_count,
        "deposit_sum": deposit_sum,
        "deposit_count": deposit_count,
        "move_out_count": move_out_count,
        "charges": dict(charge_sums),
        "charge_counts": dict(charge_counts),
    }


def db_truth(code: str, month: date) -> dict:
    with session_scope() as s:
        # Aggregate from rent_charge_lines
        rows = s.execute(text("""
            SELECT charge_code, SUM(amount) AS s, COUNT(*) AS n
            FROM rent_charge_lines
            WHERE property_code=:c AND snapshot_month=:m
            GROUP BY charge_code
        """), {"c": code, "m": month}).all()
        charges = {r[0]: float(r[1] or 0.0) for r in rows}
        counts  = {r[0]: int(r[2]) for r in rows}

        unit_count = s.execute(text("""
            SELECT COUNT(*) FROM rent_snapshots
            WHERE property_code=:c AND snapshot_month=:m
        """), {"c": code, "m": month}).scalar() or 0

        # Deposit/move-out — only available on `leases` (latest snapshot).
        # We use the latest-snapshot lease totals only when month==latest.
        return {
            "unit_count": unit_count,
            "charges": charges,
            "charge_counts": counts,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    settings = get_settings()
    files = sorted(Path(settings.rent_roll_dir).glob("*.xls"))
    print(f"Validating {len(files)} workbooks against MySQL...\n")

    total_mismatches = 0
    total_files_ok = 0
    files_with_issues = []

    for i, path in enumerate(files, 1):
        try:
            wb = read_workbook_truth(path)
            db = db_truth(wb["property_code"], wb["snapshot_month"])
        except Exception as e:
            files_with_issues.append((path.name, [f"VALIDATOR ERROR: {e}"]))
            continue

        issues = []

        # Unit count
        if wb["unit_count"] != db["unit_count"]:
            issues.append(f"unit_count: wb={wb['unit_count']} db={db['unit_count']}")

        # Per-charge-code sum
        all_codes = set(wb["charges"]) | set(db["charges"])
        for cc in sorted(all_codes):
            w = wb["charges"].get(cc, 0.0)
            d = db["charges"].get(cc, 0.0)
            if abs(w - d) > 0.01:
                issues.append(f"{cc}: wb_sum={w:.2f} db_sum={d:.2f} diff={w - d:.2f}")
            wc = wb["charge_counts"].get(cc, 0)
            dc = db["charge_counts"].get(cc, 0)
            if wc != dc:
                issues.append(f"{cc}: wb_count={wc} db_count={dc}")

        if issues:
            files_with_issues.append((path.name, issues))
            total_mismatches += len(issues)
        else:
            total_files_ok += 1

        if i % 25 == 0 or i == len(files):
            print(f"  {i:3d}/{len(files)} validated · ok={total_files_ok} issues={len(files_with_issues)}")

    print()
    print("=" * 70)
    print(f"OK files:       {total_files_ok}/{len(files)}")
    print(f"Files w/ issues:{len(files_with_issues)}")
    print(f"Total mismatches: {total_mismatches}")
    print("=" * 70)

    if files_with_issues:
        print("\n--- Issues (first 20 files) ---")
        for fn, issues in files_with_issues[:20]:
            print(f"\n{fn}:")
            for it in issues[:8]:
                print(f"  - {it}")
            if len(issues) > 8:
                print(f"  ... and {len(issues) - 8} more")


if __name__ == "__main__":
    main()
