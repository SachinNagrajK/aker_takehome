"""OpenTelemetry / Phoenix Cloud tracing bootstrap.

`init_tracing()` is called once from `main.py:lifespan`. It is **fail-open** —
any error (missing key, import problem, network issue) is logged and swallowed
so the API keeps serving traffic.

Span export uses OTel's `BatchSpanProcessor`, which queues spans in memory and
ships them on a background thread, so `/chat` request latency is unaffected by
the network or Phoenix availability.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger("property_ai.observability")

_initialized = False
_tracer_provider = None


def init_tracing(app: "FastAPI | None" = None) -> None:
    """Idempotent. Wire Phoenix Cloud + OpenInference + FastAPI instrumentation.

    Defaults to no-op when `PHOENIX_ENABLED` is false or the API key is missing.
    """
    global _initialized, _tracer_provider
    if _initialized:
        return

    from .config import get_settings
    settings = get_settings()

    if not settings.phoenix_enabled:
        log.info("phoenix tracing disabled (PHOENIX_ENABLED=false)")
        _initialized = True
        return

    if not settings.phoenix_api_key:
        log.warning("PHOENIX_ENABLED=true but PHOENIX_API_KEY is unset — tracing disabled")
        _initialized = True
        return

    # Suppress Authorization / X-Admin-Token in FastAPI request-header spans.
    os.environ.setdefault(
        "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SANITIZE_FIELDS",
        "authorization,x-admin-token,cookie,set-cookie",
    )

    try:
        from phoenix.otel import register

        # Phoenix Cloud spaces require Authorization: Bearer <key>. We set it
        # both via the register() kwarg and as OTEL_EXPORTER_OTLP_HEADERS so
        # whichever transport phoenix.otel picks (gRPC or HTTP) authenticates.
        bearer = f"Bearer {settings.phoenix_api_key}"
        os.environ.setdefault("OTEL_EXPORTER_OTLP_HEADERS", f"Authorization={bearer}")
        os.environ.setdefault("PHOENIX_API_KEY", settings.phoenix_api_key)

        _tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_endpoint,
            headers={"Authorization": bearer},
            batch=True,
            auto_instrument=False,
            set_global_tracer_provider=True,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("phoenix.otel.register failed — tracing disabled: %s", e)
        _initialized = True
        return

    # Instrument LangChain (covers LangGraph nodes), LLM SDKs, and FastAPI.
    # Each is wrapped independently so a missing dep doesn't kill the others.
    _safe_instrument("openinference.instrumentation.langchain", "LangChainInstrumentor")
    _safe_instrument("openinference.instrumentation.openai", "OpenAIInstrumentor")
    _safe_instrument("openinference.instrumentation.anthropic", "AnthropicInstrumentor")
    _safe_instrument("openinference.instrumentation.google_genai", "GoogleGenAIInstrumentor")

    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            FastAPIInstrumentor.instrument_app(app)
        except Exception as e:  # noqa: BLE001
            log.warning("FastAPI instrumentation failed: %s", e)

    log.info(
        "phoenix tracing enabled — project=%s endpoint=%s",
        settings.phoenix_project_name,
        settings.phoenix_endpoint,
    )
    _initialized = True


def _safe_instrument(module: str, class_name: str) -> None:
    try:
        mod = __import__(module, fromlist=[class_name])
        cls = getattr(mod, class_name)
        cls().instrument()
    except Exception as e:  # noqa: BLE001
        log.warning("instrumentor %s.%s skipped: %s", module, class_name, e)


def shutdown_tracing() -> None:
    """Flush in-flight spans on app shutdown."""
    global _tracer_provider
    if _tracer_provider is None:
        return
    try:
        _tracer_provider.shutdown()
    except Exception as e:  # noqa: BLE001
        log.warning("tracer provider shutdown failed: %s", e)
    _tracer_provider = None


def get_tracer(name: str = "property_ai"):
    """Return an OTel tracer. Safe to call even when tracing is disabled — the
    no-op TracerProvider returns no-op spans."""
    from opentelemetry import trace
    return trace.get_tracer(name)
