"""APScheduler-based periodic eval runner.

Runs in the FastAPI process on a separate thread pool (max_workers=1,
coalesce=True, max_instances=1) so overlapping runs can't pile up and the
request path is never affected. Opt-in via EVAL_SCHEDULE_ENABLED.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from ..config import get_settings

log = logging.getLogger("property_ai.evals.scheduler")

_scheduler: Any = None
_job_id = "eval_golden_set"


def start_scheduler() -> None:
    global _scheduler
    settings = get_settings()
    if not settings.eval_schedule_enabled:
        log.info("eval scheduler disabled (EVAL_SCHEDULE_ENABLED=false)")
        return
    if _scheduler is not None:
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.executors.pool import ThreadPoolExecutor
    except Exception as e:  # noqa: BLE001
        log.warning("APScheduler not available; scheduling disabled: %s", e)
        return

    _scheduler = BackgroundScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=1)},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 300},
        timezone="UTC",
    )
    try:
        trigger = CronTrigger.from_crontab(settings.eval_schedule_cron, timezone="UTC")
    except Exception as e:  # noqa: BLE001
        log.warning("invalid EVAL_SCHEDULE_CRON=%r (%s); scheduler not started", settings.eval_schedule_cron, e)
        _scheduler = None
        return

    _scheduler.add_job(_run_scheduled, trigger=trigger, id=_job_id, replace_existing=True)
    _scheduler.start()
    log.info("eval scheduler started — cron=%r", settings.eval_schedule_cron)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception as e:  # noqa: BLE001
        log.warning("scheduler shutdown error: %s", e)
    _scheduler = None


def get_status() -> dict[str, Any]:
    settings = get_settings()
    status: dict[str, Any] = {
        "enabled": settings.eval_schedule_enabled,
        "cron": settings.eval_schedule_cron,
        "running": _scheduler is not None,
        "next_run_at": None,
    }
    if _scheduler is not None:
        job = _scheduler.get_job(_job_id)
        if job and job.next_run_time:
            status["next_run_at"] = job.next_run_time.isoformat()
    return status


def update_cron(cron: str) -> dict[str, Any]:
    """Reschedule the existing job to a new crontab string. Returns status."""
    if _scheduler is None:
        raise RuntimeError("scheduler not running")
    from apscheduler.triggers.cron import CronTrigger
    trigger = CronTrigger.from_crontab(cron, timezone="UTC")
    _scheduler.reschedule_job(_job_id, trigger=trigger)
    # Mutate the in-memory setting too (process-local; not persisted to .env).
    get_settings().eval_schedule_cron = cron
    return get_status()


def _run_scheduled() -> None:
    # Imported lazily so a scheduler tick can never fail at module-import time.
    from . import runner
    try:
        runner.run_eval(run_id=str(uuid.uuid4()), trigger="scheduled")
    except Exception as e:  # noqa: BLE001
        log.exception("scheduled eval run failed: %s", e)
