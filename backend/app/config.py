"""Central config + LLM model registry."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend/ regardless of CWD.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")


class Settings:
    # DB — Postgres (Supabase in prod, docker-compose locally).
    # Canonical: full SQLAlchemy URL via DATABASE_URL / DATABASE_READER_URL.
    # Fallback: build from PG_* parts for local docker-compose convenience.
    database_url: str = (
        os.getenv("DATABASE_URL")
        or "postgresql+psycopg://{u}:{p}@{h}:{port}/{db}".format(
            u=os.getenv("PG_USER", "property_user"),
            p=os.getenv("PG_PASSWORD", "property_pass"),
            h=os.getenv("PG_HOST", "localhost"),
            port=os.getenv("PG_PORT", "5432"),
            db=os.getenv("PG_DB", "property_ai"),
        )
    )

    # Read-only role — used only by the LLM-written SQL executor.
    # Lacks INSERT/UPDATE/DELETE/DDL privileges at the Postgres level even if
    # the sqlglot validator is bypassed.
    database_reader_url: str = (
        os.getenv("DATABASE_READER_URL")
        or "postgresql+psycopg://{u}:{p}@{h}:{port}/{db}".format(
            u=os.getenv("PG_READER_USER", "property_reader"),
            p=os.getenv("PG_READER_PASSWORD", "reader_pass"),
            h=os.getenv("PG_HOST", "localhost"),
            port=os.getenv("PG_PORT", "5432"),
            db=os.getenv("PG_DB", "property_ai"),
        )
    )

    # LLM keys
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None

    # Vector store — Pinecone serverless. Each property is a namespace so
    # retrieval doesn't need a property_code metadata filter.
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY") or None
    pinecone_index: str = os.getenv("PINECONE_INDEX", "property-chunks-v2")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    doc_store_dir: str = os.getenv("DOC_STORE_DIR", str(_BACKEND_DIR / "doc_store"))
    # Image/table artifacts move to Supabase Storage in production; the local
    # doc_store dir stays as an ingestion-time cache only.
    supabase_url: str | None = os.getenv("SUPABASE_URL") or None
    supabase_service_role_key: str | None = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None
    supabase_storage_bucket: str = os.getenv("SUPABASE_STORAGE_BUCKET", "doc-store")
    embedding_model_v2: str = os.getenv("EMBEDDING_MODEL_V2", "jinaai/jina-clip-v2")
    embedding_quant: str = os.getenv("EMBEDDING_QUANT", "int8")  # int8 | fp32 (ONNX runtime)
    admin_token: str | None = os.getenv("ADMIN_TOKEN") or None

    # Data
    rent_roll_dir: str = os.getenv(
        "RENT_ROLL_DIR",
        str(_BACKEND_DIR.parent.parent / "RentRoll_LeaseCharges_NamesRedacted" / "RentRoll_LeaseCharges_NamesRedacted"),
    )

    # Server
    backend_host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))

    # Observability — Phoenix Cloud (hosted, free tier). Defaults off so the
    # app still boots when keys aren't set. Set PHOENIX_ENABLED=true and
    # PHOENIX_API_KEY in prod env. Export is non-blocking (BatchSpanProcessor
    # ships spans from a background thread).
    phoenix_enabled: bool = (os.getenv("PHOENIX_ENABLED", "false").strip().lower() == "true")
    phoenix_endpoint: str = os.getenv("PHOENIX_ENDPOINT", "https://app.phoenix.arize.com/s/aker-ai/v1/traces")
    phoenix_api_key: str | None = os.getenv("PHOENIX_API_KEY") or None
    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "property-ai")

    # Evaluation harness — never runs inline on /chat. Manual via UI/API,
    # or scheduled via APScheduler. open_rag_eval judge model.
    eval_judge_model: str = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    eval_schedule_enabled: bool = (os.getenv("EVAL_SCHEDULE_ENABLED", "false").strip().lower() == "true")
    eval_schedule_cron: str = os.getenv("EVAL_SCHEDULE_CRON", "0 */6 * * *")
    eval_max_cases: int = int(os.getenv("EVAL_MAX_CASES", "50"))
    # Eval runs are persisted to Supabase Postgres (via SQLAlchemy / app.db).
    # Local JSONL snapshots are still written for grep-friendly debugging.
    eval_results_dir: str = os.getenv("EVAL_RESULTS_DIR", str(_BACKEND_DIR / "evals" / "results"))

    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url

    @property
    def sqlalchemy_reader_url(self) -> str:
        """Connection string for the read-only DB user."""
        return self.database_reader_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# LLM model registry. Surfaced via GET /llms.
MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-haiku-4-5", "claude-sonnet-4-5"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-flash"],
}


def available_providers() -> dict[str, bool]:
    s = get_settings()
    return {
        "openai": bool(s.openai_api_key),
        "anthropic": bool(s.anthropic_api_key),
        "gemini": bool(s.google_api_key),
    }


# Mapping of property_code -> list of marketing URLs to scrape.
#
# NOTE: The rent-roll properties (e.g. 115r = "Canfield Park", 126r = "The Halden")
# don't actually correspond to the marketing sites below. We deliberately map two
# real codes to the user-provided marketing URLs purely so the RAG pipeline has
# representative unstructured content to demonstrate property-scoped retrieval.
# Documented as an assumption in README.md.
SCRAPE_SEEDS: dict[str, list[str]] = {
    "115r": [
        "https://albanywatersview.com/",
        "https://albanywatersview.com/amenities/",
        "https://albanywatersview.com/floorplans/",
    ],
    "126r": [
        "https://thehamletatsaratogasprings.com/",
        "https://thehamletatsaratogasprings.com/amenities/",
        "https://thehamletatsaratogasprings.com/floorplans/",
    ],
}
