"""FastAPI entry point.

Routes:
  GET  /health       - liveness probe
  GET  /properties   - list all property codes + names
  GET  /llms         - list providers + models + availability
  POST /chat         - run the LangGraph orchestrator scoped to a property

Errors are translated centrally:
  UnknownPropertyError -> 404
  ScopeViolationError  -> 400
  ProviderUnavailable  -> 400
  ValueError           -> 400
  Anything else        -> 500
"""
from __future__ import annotations

import logging
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from .config import get_settings
from .db import init_db, session_scope
from .models import Property
from .schemas import (
    ChatRequest, ChatResponse, Clarification, LLMOption,
    PropertyOut, Source, ToolTraceStep, UIComponent,
)
from .guardrails.scope import UnknownPropertyError, ScopeViolationError
from .llm_registry import ProviderUnavailable, list_llms, validate_model
from .graph.build import run_chat

log = logging.getLogger("property_ai")
settings = get_settings()

app = FastAPI(title="Property-Specific AI Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@app.get("/properties", response_model=list[PropertyOut])
def list_properties() -> list[PropertyOut]:
    with session_scope() as s:
        rows = s.execute(
            select(Property.property_code, Property.property_name, Property.property_type)
            .order_by(Property.property_code)
        ).all()
    return [
        PropertyOut(property_code=r[0], property_name=r[1], property_type=r[2])
        for r in rows
    ]


# ---------------------------------------------------------------------------
# LLM manifest
# ---------------------------------------------------------------------------

@app.get("/llms", response_model=list[LLMOption])
def get_llms() -> list[LLMOption]:
    return [LLMOption(**entry) for entry in list_llms()]


# ---------------------------------------------------------------------------
# Chat — runs the LangGraph orchestrator
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Pre-validate (provider, model) before invoking the graph so the user
    # gets a clean 400 instead of a 500 from inside a node.
    try:
        validate_model(req.llm_provider, req.model)
    except ProviderUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e))

    conversation_id = req.conversation_id or str(uuid.uuid4())

    try:
        if req.clarification_reply is not None:
            # Resume an interrupted graph run with the user's chosen scope.
            state = run_chat(
                property_code=req.property_code,
                user_message=req.message,
                llm_provider=req.llm_provider,
                model=req.model,
                conversation_id=conversation_id,
                resume_value=req.clarification_reply,
            )
        else:
            state = run_chat(
                property_code=req.property_code,
                user_message=req.message,
                llm_provider=req.llm_provider,
                model=req.model,
                conversation_id=conversation_id,
            )
    except UnknownPropertyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ScopeViolationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProviderUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("chat pipeline failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Paused at a clarification interrupt — surface that to the frontend.
    if state.get("paused"):
        clar = state.get("clarification") or {}
        return ChatResponse(
            scope_kind=(state.get("scope") or {}).get("kind", "missing"),
            scope_enforced=True,
            answer_markdown="",
            llm={"provider": req.llm_provider, "model": req.model},
            clarification=Clarification(
                question=clar.get("question", "Which property?"),
                options=clar.get("options") or [],
                scope_kind=clar.get("scope_kind"),
            ),
            conversation_id=conversation_id,
        )

    scope = state.get("scope") or {}
    tool_history = state.get("tool_history") or []

    return ChatResponse(
        property_code=scope.get("code"),
        property_codes=scope.get("codes") or [],
        scope_kind=scope.get("kind", "single"),
        scope_source=scope.get("source"),
        scope_enforced=True,
        answer_markdown=state.get("answer_markdown", ""),
        components=[UIComponent(**c) for c in state.get("components", [])],
        sources=[Source(**s) for s in state.get("sources", [])],
        tool_trace=[
            ToolTraceStep(
                tool=s.get("tool"),
                args=s.get("args") or {},
                ok=bool(s.get("ok")),
                error=s.get("error"),
                duration_ms=s.get("duration_ms"),
            )
            for s in tool_history
        ],
        route=state.get("route", "agent"),
        gave_up=bool(state.get("gave_up")),
        llm={"provider": req.llm_provider, "model": req.model},
        conversation_id=conversation_id,
    )
