"""Persistence for eval runs — backed by Supabase Postgres via SQLAlchemy.

Tables (`eval_runs`, `eval_cases`) live alongside the rent-roll schema and are
created automatically by `init_db()` in `app/db.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, desc

from ..db import session_scope
from ..models import EvalCase, EvalRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_run(run_id: str, trigger: str) -> None:
    with session_scope() as s:
        # Idempotent — the API may pre-create the row before BackgroundTasks
        # hands off to the runner.
        existing = s.get(EvalRun, run_id)
        if existing is not None:
            return
        s.add(EvalRun(
            id=run_id,
            started_at=_now(),
            trigger=trigger,
            status="running",
            summary=None,
        ))


def finish_run(run_id: str, status: str, summary: dict[str, Any]) -> None:
    with session_scope() as s:
        row = s.get(EvalRun, run_id)
        if row is None:
            row = EvalRun(id=run_id, started_at=_now(), trigger="unknown", status=status, summary=summary)
            s.add(row)
        row.finished_at = _now()
        row.status = status
        row.summary = summary


def add_case(
    *,
    run_id: str,
    golden_id: str,
    property_code: str | None,
    question: str,
    answer: str | None,
    scores: dict[str, Any] | None,
    ok: bool,
    error: str | None,
    duration_ms: int | None,
    trace_id: str | None,
) -> None:
    with session_scope() as s:
        s.add(EvalCase(
            run_id=run_id,
            golden_id=golden_id,
            property_code=property_code,
            question=question,
            answer=answer,
            scores=scores,
            ok=bool(ok),
            error=error,
            duration_ms=duration_ms,
            trace_id=trace_id,
        ))


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    with session_scope() as s:
        rows = s.execute(
            select(EvalRun).order_by(desc(EvalRun.started_at)).limit(limit)
        ).scalars().all()
        return [_run_to_dict(r) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with session_scope() as s:
        run = s.get(EvalRun, run_id)
        if run is None:
            return None
        cases = s.execute(
            select(EvalCase).where(EvalCase.run_id == run_id).order_by(EvalCase.id)
        ).scalars().all()
        out = _run_to_dict(run)
        out["cases"] = [_case_to_dict(c) for c in cases]
        return out


def _run_to_dict(r: EvalRun) -> dict[str, Any]:
    return {
        "id": r.id,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "trigger": r.trigger,
        "status": r.status,
        "summary": r.summary,
    }


def _case_to_dict(c: EvalCase) -> dict[str, Any]:
    return {
        "golden_id": c.golden_id,
        "property_code": c.property_code,
        "question": c.question,
        "answer": c.answer,
        "scores": c.scores,
        "ok": bool(c.ok),
        "error": c.error,
        "duration_ms": c.duration_ms,
        "trace_id": c.trace_id,
    }
