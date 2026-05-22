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

import json
import logging
import uuid
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select

from .config import SCRAPE_SEEDS, get_settings
from .db import init_db, session_scope
from .models import Property
from .schemas import (
    ChatRequest, ChatResponse, Clarification, LLMOption,
    PropertyOut, Source, ToolTraceStep, UIComponent,
)
from .guardrails.scope import UnknownPropertyError, ScopeViolationError
from .llm_registry import ProviderUnavailable, list_llms, validate_model
from .graph.build import run_chat, run_chat_stream

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

# Mount the local image/table doc store for v2 RAG so the frontend can
# load `<img src="/doc_store/{property}/{hash}.png" />`. Created up-front
# so StaticFiles doesn't reject a missing dir at import time.
_doc_root = Path(settings.doc_store_dir)
_doc_root.mkdir(parents=True, exist_ok=True)
app.mount("/doc_store", StaticFiles(directory=str(_doc_root)), name="doc_store")


@app.on_event("startup")
def _startup() -> None:
    init_db()


# ---------------------------------------------------------------------------
# Admin — RAG v2 ingestion
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    property_code: str = Field(min_length=1)
    urls: list[str] = Field(default_factory=list)
    replace: bool = True


def _require_admin(token: str | None) -> None:
    expected = settings.admin_token
    if not expected:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured on server")
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.get("/admin/ingest/defaults")
def admin_ingest_defaults(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    _require_admin(x_admin_token)
    return {"seeds": SCRAPE_SEEDS}


@app.post("/admin/ingest")
def admin_ingest(
    req: IngestRequest,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> dict:
    _require_admin(x_admin_token)
    if not req.urls:
        raise HTTPException(status_code=400, detail="urls must be non-empty")
    # Import lazily so the heavy model load (transformers/docling) only
    # happens when an admin actually triggers ingestion.
    from .ingestion.v2.pipeline import ingest_urls
    try:
        return ingest_urls(req.property_code, req.urls, replace=req.replace)
    except Exception as e:  # noqa: BLE001
        log.exception("ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


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


# ---------------------------------------------------------------------------
# Streaming chat — Server-Sent Events
# ---------------------------------------------------------------------------
#
# Each event is one line of `data: {json}\n\n`. The frontend reads lines via
# fetch+ReadableStream (EventSource doesn't support POST). Events are the
# same shape `run_chat_stream` yields — see its docstring for the discriminator.

def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n".encode("utf-8")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    # Pre-validate the LLM choice — emit one error event then close.
    try:
        validate_model(req.llm_provider, req.model)
    except ProviderUnavailable as e:
        async def _err():
            yield _sse({"type": "error", "message": str(e)})
        return StreamingResponse(_err(), media_type="text/event-stream")

    conversation_id = req.conversation_id or str(uuid.uuid4())

    async def gen():
        # The conversation_id goes out on the very first event so the client
        # can stash it before any errors.
        yield _sse({"type": "open", "conversation_id": conversation_id})
        try:
            async for event in run_chat_stream(
                property_code=req.property_code,
                user_message=req.message,
                llm_provider=req.llm_provider,
                model=req.model,
                conversation_id=conversation_id,
                resume_value=req.clarification_reply,
            ):
                # Inject conversation_id into terminal events so the client
                # can always reconcile when it sees them.
                if event.get("type") in {"done", "clarification"}:
                    event = {**event, "conversation_id": conversation_id}
                yield _sse(event)
        except UnknownPropertyError as e:
            yield _sse({"type": "error", "message": str(e), "code": 404})
        except (ScopeViolationError, ProviderUnavailable, ValueError) as e:
            yield _sse({"type": "error", "message": str(e), "code": 400})
        except Exception as e:  # noqa: BLE001
            log.exception("chat stream failed")
            yield _sse({"type": "error", "message": f"Internal error: {e}", "code": 500})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # tell nginx/proxies not to buffer
            "Connection": "keep-alive",
        },
    )
