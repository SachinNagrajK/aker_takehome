"""ORM models for the rent-roll domain."""
from __future__ import annotations

from datetime import date
from sqlalchemy import (
    String, Integer, Float, Date, Boolean, ForeignKey, JSON, Index
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    raw_row: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    property: Mapped[Property] = relationship(back_populates="snapshots")

    __table_args__ = (
        Index("ix_snap_code_month_unit", "property_code", "snapshot_month", "unit_number"),
    )
