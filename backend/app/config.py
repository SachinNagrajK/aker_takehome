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
    # DB
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "property_user")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "property_pass")
    mysql_db: str = os.getenv("MYSQL_DB", "property_ai")

    # Read-only DB user — used only by the LLM-written SQL executor.
    # Lacks INSERT/UPDATE/DELETE/DDL privileges at the MySQL level even if
    # the sqlglot validator is bypassed.
    mysql_reader_user: str = os.getenv("MYSQL_READER_USER", "property_reader")
    mysql_reader_password: str = os.getenv("MYSQL_READER_PASSWORD", "reader_pass")

    # LLM keys
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or None

    # Vector store (v1 — legacy, kept for rollback)
    chroma_dir: str = os.getenv("CHROMA_DIR", str(_BACKEND_DIR / "chroma_db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Vector store (v2 — docling + Jina v4 multimodal)
    rag_version: str = os.getenv("RAG_VERSION", "v2").lower()
    chroma_dir_v2: str = os.getenv("CHROMA_DIR_V2", str(_BACKEND_DIR / "chroma_db_v2"))
    collection_v2: str = os.getenv("COLLECTION_V2", "property_chunks_v2")
    doc_store_dir: str = os.getenv("DOC_STORE_DIR", str(_BACKEND_DIR / "doc_store"))
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

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )

    @property
    def sqlalchemy_reader_url(self) -> str:
        """Connection string for the read-only DB user."""
        return (
            f"mysql+pymysql://{self.mysql_reader_user}:{self.mysql_reader_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )


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
