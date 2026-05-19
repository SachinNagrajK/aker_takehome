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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from .config import get_settings
from .db import init_db, session_scope
from .models import Property
from .schemas import ChatRequest, ChatResponse, LLMOption, PropertyOut, Source, UIComponent
from .guardrails.scope import UnknownPropertyError, ScopeViolationError
from .llm.base import ProviderUnavailable
from .llm.factory import list_llms, validate_model
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

    try:
        state = run_chat(
            property_code=req.property_code,
            user_message=req.message,
            llm_provider=req.llm_provider,
            model=req.model,
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

    return ChatResponse(
        property_code=state["property_code"],
        scope_enforced=True,
        answer_markdown=state.get("answer_markdown", ""),
        components=[UIComponent(**c) for c in state.get("components", [])],
        sources=[Source(**s) for s in state.get("sources", [])],
        route=state.get("route", "sql"),
        llm={"provider": req.llm_provider, "model": req.model},
    )
