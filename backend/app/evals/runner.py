"""Run the golden set against the live graph and score each turn.

Sync function — safe to call from a background thread (APScheduler) or from
FastAPI's `BackgroundTasks`. NEVER call from within a `/chat` request path.

Phoenix Cloud spans are emitted naturally by the LangChain instrumentor —
each case becomes its own trace because we pass a fresh `conversation_id`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..config import get_settings
from ..graph.build import run_chat
from ..observability import get_tracer
from . import scorer, store

log = logging.getLogger("property_ai.evals.runner")
_tracer = get_tracer("property_ai.evals")

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.yaml"


def load_golden() -> list[dict[str, Any]]:
    with _GOLDEN_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("cases") or [])


def run_eval(
    *,
    run_id: str | None = None,
    golden_ids: list[str] | None = None,
    trigger: str = "manual",
    llm_provider: str = "openai",
    model: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    cases = load_golden()
    if golden_ids:
        wanted = set(golden_ids)
        cases = [c for c in cases if c.get("id") in wanted]
    if not cases:
        return {"run_id": run_id, "status": "empty", "summary": {"count": 0}}

    cases = cases[: settings.eval_max_cases]
    run_id = run_id or str(uuid.uuid4())
    model = model or settings.eval_judge_model

    store.create_run(run_id, trigger)
    log.info("eval run %s started (trigger=%s, cases=%d)", run_id, trigger, len(cases))

    results: list[dict[str, Any]] = []

    with _tracer.start_as_current_span("eval.run") as run_span:
        run_span.set_attribute("eval.run_id", run_id)
        run_span.set_attribute("eval.trigger", trigger)
        run_span.set_attribute("eval.case_count", len(cases))

        for case in cases:
            case_result = _run_one_case(run_id=run_id, case=case, provider=llm_provider, model=model)
            results.append(case_result)
            store.add_case(
                run_id=run_id,
                golden_id=case.get("id", ""),
                property_code=case.get("property_code"),
                question=case.get("question", ""),
                answer=case_result.get("answer"),
                scores=case_result.get("scores"),
                ok=case_result.get("ok", False),
                error=case_result.get("error"),
                duration_ms=case_result.get("duration_ms"),
                trace_id=case_result.get("trace_id"),
            )

    summary = _summarize(results)
    store.finish_run(run_id, status="completed", summary=summary)
    _write_jsonl(run_id, results, summary)
    log.info("eval run %s finished — %s", run_id, summary)
    return {"run_id": run_id, "status": "completed", "summary": summary}


def _run_one_case(*, run_id: str, case: dict[str, Any], provider: str, model: str) -> dict[str, Any]:
    case_id = case.get("id", "")
    question = case.get("question", "")
    property_code = case.get("property_code")
    started = time.monotonic()

    answer: str | None = None
    tool_history: list[dict] = []
    err: str | None = None
    trace_id: str | None = None

    with _tracer.start_as_current_span("eval.case") as span:
        span.set_attribute("eval.case_id", case_id)
        span.set_attribute("eval.property_code", property_code or "")
        try:
            ctx = span.get_span_context()
            trace_id = format(ctx.trace_id, "032x") if ctx and ctx.trace_id else None
        except Exception:  # noqa: BLE001
            trace_id = None

        try:
            conv_id = f"eval-{run_id}-{case_id}"
            state = run_chat(
                property_code=property_code,
                user_message=question,
                llm_provider=provider,
                model=model,
                conversation_id=conv_id,
            )
            # The graph may pause for property OR time-scope clarification.
            # Auto-resume with sensible defaults so a single eval case
            # exercises the full agent loop end-to-end:
            #   - time clarification → "Latest"
            #   - property clarification → first option (or skip-score it later)
            # Hard cap of 2 resumes to avoid loops.
            for _ in range(2):
                if not state.get("paused"):
                    break
                clar = state.get("clarification") or {}
                kind = clar.get("scope_kind")
                options = clar.get("options") or []
                if kind == "time":
                    choice = "Latest"
                elif options:
                    choice = options[0]
                else:
                    break
                state = run_chat(
                    property_code=property_code,
                    user_message=question,
                    llm_provider=provider,
                    model=model,
                    conversation_id=conv_id,
                    resume_value=choice,
                )
            answer = state.get("answer_markdown") or ""
            tool_history = state.get("tool_history") or []
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            log.exception("eval case %s failed in graph", case_id)

        scores: dict[str, Any] | None = None
        if err is None:
            contexts = scorer.contexts_from_tool_history(tool_history)
            scores = scorer.score_turn(question, answer or "", contexts)
            for k in ("groundedness", "hallucination", "answer_relevance", "context_relevance"):
                v = scores.get(k)
                if isinstance(v, (int, float)):
                    span.set_attribute(f"eval.{k}", float(v))

    return {
        "case_id": case_id,
        "property_code": property_code,
        "question": question,
        "answer": answer,
        "scores": scores,
        "ok": err is None,
        "error": err,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "trace_id": trace_id,
        "tool_calls": [s.get("tool") for s in tool_history if isinstance(s, dict)],
        "expected_tools": case.get("expected_tools") or [],
        "expected_substrings": case.get("expected_substrings") or [],
        "substring_hits": _substring_hits(answer or "", case.get("expected_substrings") or []),
    }


def _substring_hits(text: str, needles: list[str]) -> int:
    t = (text or "").lower()
    return sum(1 for n in needles if n and n.lower() in t)


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"count": 0}
    ok_n = sum(1 for r in results if r.get("ok"))
    out: dict[str, Any] = {
        "count": n,
        "ok_count": ok_n,
        "error_count": n - ok_n,
    }
    for k in ("groundedness", "hallucination", "answer_relevance", "context_relevance"):
        vals = [(r.get("scores") or {}).get(k) for r in results]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        if vals:
            out[f"mean_{k}"] = round(sum(vals) / len(vals), 3)
            out[f"min_{k}"] = round(min(vals), 3)
    return out


def _write_jsonl(run_id: str, results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir = Path(get_settings().eval_results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}_{run_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_summary": summary, "run_id": run_id}, default=str) + "\n")
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run the RAG eval golden set.")
    ap.add_argument("--ids", help="Comma-separated golden case IDs (default: all).")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    ids = [s.strip() for s in (args.ids or "").split(",") if s.strip()] or None
    out = run_eval(golden_ids=ids, trigger="cli", llm_provider=args.provider, model=args.model)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    _main()
