"""Drop all rent-roll tables, recreate the schema, and re-ingest every
monthly workbook from the directory in RENT_ROLL_DIR.

Use after editing models.py — `init_db()` (create_all) cannot ALTER existing
tables to add new columns, so the only safe upgrade path is drop+recreate.

  python wipe_and_reingest.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import text
from app.db import engine, init_db
from app.config import get_settings
from app.ingestion.rent_roll import ingest_directory


# Order matters — FK dependencies first.
DROP_ORDER = [
    "rent_charge_lines",
    "rent_snapshots",
    "leases",
    "units",
    "properties",
]


def main() -> None:
    settings = get_settings()
    t0 = time.time()

    print("[1/3] Dropping rent-roll tables ...")
    with engine.begin() as conn:
        # MySQL refuses to drop FK parents while children exist; disabling FK
        # checks for the duration of the wipe is simpler than dependency math.
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for tbl in DROP_ORDER:
            conn.execute(text(f"DROP TABLE IF EXISTS `{tbl}`"))
            print(f"    dropped {tbl}")
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

    print("[2/3] Recreating schema (with v4 columns: resident_deposit, "
          "other_deposit, move_out_date on leases) ...")
    init_db()

    print(f"[3/3] Ingesting workbooks from {settings.rent_roll_dir} ...")
    ingest_directory(Path(settings.rent_roll_dir))

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
