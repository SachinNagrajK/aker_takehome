"""SQLAlchemy engine, session, and DDL bootstrap."""
from __future__ import annotations

from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .config import get_settings

_settings = get_settings()
engine = create_engine(_settings.sqlalchemy_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope() -> Session:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    from . import models  # noqa: F401  (register mappers)
    models.Base.metadata.create_all(engine)
