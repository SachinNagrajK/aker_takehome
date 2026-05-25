"""ORM models for the rent-roll domain."""
from __future__ import annotations

from datetime import date, datetime
from sqlalchemy import (
    String, Integer, Float, Date, Boolean, ForeignKey, JSON, Index, DateTime, Text
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# Use JSONB on Postgres for indexed `->`/`->>` access on raw_row; fall back
# to generic JSON on other dialects (SQLite tests, MySQL, etc.).
JSON_VARIANT = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Property(Base):
    __tablename__ = "properties"
    property_code: Mapped[str] = mapped_column(String(32), primary_key=True)
    property_name: Mapped[str] = mapped_column(String(255))
    property_type: Mapped[str] = mapped_column(String(32))   # r / a / c / land / other
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)

    units: Mapped[list["Unit"]] = relationship(back_populates="property", cascade="all, delete-orphan")
    leases: Mapped[list["Lease"]] = relationship(back_populates="property", cascade="all, delete-orphan")
    snapshots: Mapped[list["RentSnapshot"]] = relationship(back_populates="property", cascade="all, delete-orphan")


class Unit(Base):
    __tablename__ = "units"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    property_code: Mapped[str] = mapped_column(String(32), ForeignKey("properties.property_code"), index=True)
    unit_number: Mapped[str] = mapped_column(String(64))
    unit_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bedrooms: Mapped[float | None] = mapped_column(Float, nullable=True)
    bathrooms: Mapped[float | None] = mapped_column(Float, nullable=True)
    sqft: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_rent: Mapped[float | None] = mapped_column(Float, nullable=True)

    property: Mapped[Property] = relationship(back_populates="units")

    __table_args__ = (
        Index("ix_units_code_unit", "property_code", "unit_number", unique=True),
    )


class Lease(Base):
    __tablename__ = "leases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    property_code: Mapped[str] = mapped_column(String(32), ForeignKey("properties.property_code"), index=True)
    unit_number: Mapped[str] = mapped_column(String(64))
    tenant_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # already redacted upstream
    lease_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    lease_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    monthly_rent: Mapped[float | None] = mapped_column(Float, nullable=True)
    balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # v4 additions — were silently dropped before:
    resident_deposit: Mapped[float | None] = mapped_column(Float, nullable=True)
    other_deposit:    Mapped[float | None] = mapped_column(Float, nullable=True)
    move_out_date:    Mapped[date | None] = mapped_column(Date, nullable=True)

    property: Mapped[Property] = relationship(back_populates="leases")


class RentSnapshot(Base):
    """One row per (property, unit, month). Powers time-series queries."""
    __tablename__ = "rent_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    property_code: Mapped[str] = mapped_column(String(32), ForeignKey("properties.property_code"), index=True)
    snapshot_month: Mapped[date] = mapped_column(Date, index=True)
    unit_number: Mapped[str] = mapped_column(String(64))
    monthly_rent: Mapped[float | None] = mapped_column(Float, nullable=True)
    occupied: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    raw_row: Mapped[dict | None] = mapped_column(JSON_VARIANT, nullable=True)

    property: Mapped[Property] = relationship(back_populates="snapshots")
    charge_lines: Mapped[list["RentChargeLine"]] = relationship(back_populates="snapshot", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_snap_code_month_unit", "property_code", "snapshot_month", "unit_number"),
    )


class RentChargeLine(Base):
    """One row per charge line within a unit block.

    Preserves the granularity that `RentSnapshot.raw_row.charges` lost — that
    dict sums charges per code, so two PARKING lines of $75 and $100 collapsed
    into a single $175. This table keeps every line item with its position
    (`line_index`) so the agent can answer "list the individual parking fees".
    """
    __tablename__ = "rent_charge_lines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(Integer, ForeignKey("rent_snapshots.id", ondelete="CASCADE"), index=True)
    property_code: Mapped[str] = mapped_column(String(32), ForeignKey("properties.property_code"), index=True)
    snapshot_month: Mapped[date] = mapped_column(Date, index=True)
    unit_number: Mapped[str] = mapped_column(String(64))
    line_index: Mapped[int] = mapped_column(Integer)
    charge_code: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    snapshot: Mapped["RentSnapshot"] = relationship(back_populates="charge_lines")

    __table_args__ = (
        Index("ix_charge_code_month_unit", "property_code", "snapshot_month", "unit_number"),
    )


# ---------------------------------------------------------------------------
# Evaluation harness — persisted alongside the rent-roll schema in Supabase
# so the Monitoring UI can read run history without a separate datastore.
# ---------------------------------------------------------------------------

class EvalRun(Base):
    __tablename__ = "eval_runs"
    id:           Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    trigger:      Mapped[str] = mapped_column(String(32))   # manual | scheduled | cli
    status:       Mapped[str] = mapped_column(String(16))   # running | completed | failed
    summary:      Mapped[dict | None] = mapped_column(JSON_VARIANT, nullable=True)

    cases: Mapped[list["EvalCase"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )


class EvalCase(Base):
    __tablename__ = "eval_cases"
    id:            Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id:        Mapped[str] = mapped_column(String(64), ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    golden_id:     Mapped[str] = mapped_column(String(128))
    property_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    question:      Mapped[str] = mapped_column(Text)
    answer:        Mapped[str | None] = mapped_column(Text, nullable=True)
    scores:        Mapped[dict | None] = mapped_column(JSON_VARIANT, nullable=True)
    ok:            Mapped[bool] = mapped_column(Boolean, default=False)
    error:         Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms:   Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace_id:      Mapped[str | None] = mapped_column(String(64), nullable=True)

    run: Mapped["EvalRun"] = relationship(back_populates="cases")
