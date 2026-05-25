"""Admin-protected FastAPI router exposing the eval harness to the UI."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from . import runner, scheduler, store

router = APIRouter(prefix="/evals", tags=["evals"])


def _require_admin(token: str | None) -> None:
    expected = get_settings().admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured on server")
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

@router.get("/golden")
def get_golden(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> list[dict]:
    _require_admin(x_admin_token)
    cases = runner.load_golden()
    return [
        {"id": c.get("id"), "question": c.get("question"), "property_code": c.get("property_code")}
        for c in cases
    ]


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    ids: list[str] | None = Field(default=None, description="Subset of golden IDs to run")
    provider: str = "openai"
    model: str | None = None


@router.post("/runs")
def trigger_run(
    req: RunRequest,
    background: BackgroundTasks,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    _require_admin(x_admin_token)
    run_id = str(uuid.uuid4())
    store.create_run(run_id, trigger="manual")
    background.add_task(
        _safe_run,
        run_id=run_id,
        golden_ids=req.ids,
        provider=req.provider,
        model=req.model,
    )
    return {"run_id": run_id, "status": "started"}


def _safe_run(*, run_id: str, golden_ids: list[str] | None, provider: str, model: str | None) -> None:
    import logging
    log = logging.getLogger("property_ai.evals.api")
    try:
        runner.run_eval(
            run_id=run_id,
            golden_ids=golden_ids,
            trigger="manual",
            llm_provider=provider,
            model=model,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("background eval run failed")
        try:
            store.finish_run(run_id, status="failed", summary={"error": f"{type(e).__name__}: {e}"})
        except Exception:  # noqa: BLE001
            pass


@router.get("/runs")
def list_runs(
    limit: int = 50,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> list[dict]:
    _require_admin(x_admin_token)
    return store.list_runs(limit=limit)


@router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    _require_admin(x_admin_token)
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class ScheduleUpdate(BaseModel):
    cron: str = Field(..., description="Crontab expression, e.g. '0 */6 * * *'")


@router.get("/schedule")
def get_schedule(x_admin_token: str | None = Header(default=None, alias="X-Admin-Token")) -> dict[str, Any]:
    _require_admin(x_admin_token)
    return scheduler.get_status()


@router.put("/schedule")
def put_schedule(
    body: ScheduleUpdate,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict[str, Any]:
    _require_admin(x_admin_token)
    try:
        return scheduler.update_cron(body.cron)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid cron: {e}")
