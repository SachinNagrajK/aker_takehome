"""RAG scoring — uses open_rag_eval when its TRECEvaluator API is available,
falls back to a direct OpenAI LLM-judge so the harness still produces numbers
when the library surface drifts between releases.

Returns a dict with four 0-1 scores: groundedness, hallucination,
answer_relevance, context_relevance.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..config import get_settings

log = logging.getLogger("property_ai.evals.scorer")


_JUDGE_SYSTEM = """You are a strict RAG evaluator. You will receive a question,
an answer the system produced, and the retrieved context passages it had
available. Score four metrics from 0.0 to 1.0 (higher is better):

- groundedness:        is every factual claim in the answer supported by the context?
- hallucination:       1.0 means NO hallucination (answer makes nothing up). 0.0 means major fabrications.
- answer_relevance:    does the answer address the question asked?
- context_relevance:   are the retrieved contexts useful for answering the question?

Reply with ONLY a compact JSON object: {"groundedness": x, "hallucination": x,
"answer_relevance": x, "context_relevance": x, "rationale": "<one short line>"}.
No prose, no markdown fences."""


def _llm_judge_score(question: str, answer: str, contexts: list[str]) -> dict[str, Any]:
    """Fallback / primary judge using OpenAI directly."""
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    ctx_blob = "\n\n---\n\n".join(contexts) if contexts else "(no contexts retrieved)"
    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"RETRIEVED CONTEXTS:\n{ctx_blob[:12000]}"
    )
    resp = client.chat.completions.create(
        model=settings.eval_judge_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    out = {
        "groundedness":      _clamp01(data.get("groundedness")),
        "hallucination":     _clamp01(data.get("hallucination")),
        "answer_relevance":  _clamp01(data.get("answer_relevance")),
        "context_relevance": _clamp01(data.get("context_relevance")),
        "rationale":         (data.get("rationale") or "")[:300],
        "judge":             f"openai/{settings.eval_judge_model}",
    }
    return out


def _try_open_rag_eval(question: str, answer: str, contexts: list[str]) -> dict[str, Any] | None:
    """Best-effort call into open_rag_eval. Returns None if the API surface
    isn't what we expect (we then fall back to direct LLM judging)."""
    try:
        from open_rag_eval.evaluators.trec_evaluator import TRECEvaluator  # type: ignore
        from open_rag_eval.models.openai_model import OpenAIModel  # type: ignore
        from open_rag_eval.rag_types import RAGResult  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.info("open_rag_eval not importable (%s); using direct judge", e)
        return None

    settings = get_settings()
    try:
        model = OpenAIModel(name=settings.eval_judge_model, api_key=settings.openai_api_key)
        evaluator = TRECEvaluator(model=model)
        result = RAGResult(
            query=question,
            generated_answer=answer,
            retrieved_contexts=contexts or [""],
        )
        scored = evaluator.evaluate_single(result)
        scores = getattr(scored, "scores", None) or {}
        # open_rag_eval metric names differ across releases — map best-effort.
        return {
            "groundedness":      _clamp01(scores.get("groundedness") or scores.get("umbrela")),
            "hallucination":     _clamp01(scores.get("hallucination") or scores.get("hhem")),
            "answer_relevance":  _clamp01(scores.get("answer_relevance")),
            "context_relevance": _clamp01(scores.get("context_relevance")),
            "rationale":         "open_rag_eval/TRECEvaluator",
            "judge":             f"open_rag_eval+{settings.eval_judge_model}",
            "raw":               {k: float(v) for k, v in scores.items() if _is_number(v)},
        }
    except Exception as e:  # noqa: BLE001
        log.warning("open_rag_eval scoring failed, falling back: %s", e)
        return None


def score_turn(question: str, answer: str, retrieved_contexts: list[str]) -> dict[str, Any]:
    """Score one RAG turn. Never raises — returns an `error` field on failure."""
    if not (get_settings().openai_api_key):
        return {"error": "OPENAI_API_KEY not set; cannot score"}
    try:
        result = _try_open_rag_eval(question, answer, retrieved_contexts)
        if result is None:
            result = _llm_judge_score(question, answer, retrieved_contexts)
        return result
    except Exception as e:  # noqa: BLE001
        log.exception("scoring failed")
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp01(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return max(0.0, min(1.0, f))


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def contexts_from_tool_history(tool_history: list[dict]) -> list[str]:
    """Extract retrieved text contexts from a graph turn's tool_history.

    RAG: pull text chunks from `search_property_pages` / `search_property_active`.
    SQL: serialize result rows as structured context so groundedness checks
    can verify numeric claims too.
    """
    out: list[str] = []
    for step in tool_history or []:
        tool = step.get("tool") or ""
        result = step.get("result") or step.get("output") or {}
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                result = {"text": result}
        if not isinstance(result, dict):
            continue
        if tool in {"search_property_pages", "search_property_active"}:
            for ch in (result.get("chunks") or []):
                t = (ch.get("text") or "").strip()
                if t:
                    out.append(t)
        else:
            # SQL/structured tool: dump up to ~2k chars as one context block.
            try:
                blob = json.dumps(result, default=str)[:2000]
            except (TypeError, ValueError):
                blob = str(result)[:2000]
            if blob:
                out.append(f"[{tool}] {blob}")
    return out
