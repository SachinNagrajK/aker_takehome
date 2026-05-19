"""Scrape marketing websites and index them in Chroma with property metadata.

For each property in `config.SCRAPE_SEEDS`:
  1. fetch each URL
  2. strip nav/script/style boilerplate -> plain text
  3. chunk into ~600-char paragraphs with ~80-char overlap
  4. embed and upsert into the `property_chunks` Chroma collection
     with metadata {property_code, url, page_title, chunk_index}

Embedding model:
  - If OPENAI_API_KEY is set    -> OpenAI text-embedding-3-small
  - Else                        -> Chroma's bundled sentence-transformers
                                   (all-MiniLM-L6-v2, downloaded on first use)

Re-runs are idempotent: existing rows for the property_code are deleted
before re-indexing.

Run:
    python -m app.ingestion.scrape_and_index
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Allow `python -m app.ingestion.scrape_and_index` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import SCRAPE_SEEDS, get_settings   # noqa: E402

_settings = get_settings()


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

USER_AGENT = (
    "Mozilla/5.0 (compatible; PropertyAIAssistant/0.1; +https://example.com)"
)
REQUEST_TIMEOUT = 15

# Tags that contribute no useful content.
STRIP_TAGS = ("script", "style", "noscript", "iframe", "svg", "form")

# These wrappers tend to hold cross-page boilerplate (nav, footer, cookie
# banners). We drop them by id/class substring match.
BOILERPLATE_SELECTORS = [
    "nav", "header", "footer",
    "[role=banner]", "[role=navigation]", "[role=contentinfo]",
    ".cookie", ".menu-toggle", ".skip-link",
]


def fetch_html(url: str) -> str:
    r = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.text


def html_to_text(html: str) -> tuple[str, str]:
    """Return (page_title, body_text)."""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""

    for tag in soup(STRIP_TAGS):
        tag.decompose()
    for sel in BOILERPLATE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    # Get text with paragraph breaks preserved.
    text = soup.get_text(separator="\n", strip=True)
    # Collapse runs of whitespace; keep paragraph breaks (\n).
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return title, text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_CHARS = 600
DEFAULT_OVERLAP = 80
MIN_CHUNK_CHARS = 80  # below this, a chunk has no useful signal — drop it.


def chunk_text(
    text: str,
    target_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Greedy paragraph packer with character overlap between chunks.

    We split on blank-line paragraphs, then pack paragraphs into ~target_chars
    windows. Overlap is added at the start of each subsequent chunk so we
    don't lose context at boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        buf, buf_len = [], 0

    for para in paragraphs:
        # If a single paragraph is huge, split it on sentence boundaries.
        if len(para) > target_chars * 1.5:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_buf, sub_len = [], 0
            for sent in sentences:
                if sub_len + len(sent) > target_chars and sub_buf:
                    buf.append(" ".join(sub_buf))
                    buf_len += sum(len(s) for s in sub_buf)
                    flush()
                    sub_buf, sub_len = [], 0
                sub_buf.append(sent)
                sub_len += len(sent)
            if sub_buf:
                buf.append(" ".join(sub_buf))
                buf_len += sub_len
            flush()
            continue

        if buf_len + len(para) > target_chars and buf:
            flush()
        buf.append(para)
        buf_len += len(para)

    flush()

    # Add overlap by prepending the tail of the previous chunk.
    if overlap and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            with_overlap.append(prev_tail + " " + chunks[i])
        chunks = with_overlap

    return chunks


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

COLLECTION_NAME = "property_chunks"


def get_collection():
    """Return (or create) the Chroma collection.

    Always uses the bundled sentence-transformers model (all-MiniLM-L6-v2).
    The OpenAI key is for the chat LLM only — embeddings stay local so the
    RAG layer has no API dependency and runs deterministically offline.

    Note: chromadb 0.5.x ships an `OpenAIEmbeddingFunction` that targets the
    deprecated openai 0.x API and crashes with openai>=1.0. Sticking to the
    default function side-steps that entirely.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    Path(_settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=_settings.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    embed_fn = DefaultEmbeddingFunction()
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return coll, "sentence-transformers:all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def _stable_chunk_id(property_code: str, url: str, idx: int) -> str:
    safe_url = re.sub(r"[^A-Za-z0-9]+", "_", urlparse(url).path or "root").strip("_") or "root"
    return f"{property_code}__{safe_url}__{idx:03d}"


def index_property(coll, property_code: str, urls: list[str]) -> int:
    """Scrape URLs for one property and write chunks to the collection."""
    # Idempotent: drop any existing chunks for this property first.
    coll.delete(where={"property_code": property_code})

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict] = []

    for url in urls:
        try:
            html = fetch_html(url)
        except Exception as e:
            print(f"   [!] {property_code} {url}: {e}")
            continue
        title, text = html_to_text(html)
        if not text:
            print(f"   [!] {property_code} {url}: empty body after stripping")
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_ids.append(_stable_chunk_id(property_code, url, i))
            all_docs.append(chunk)
            all_meta.append({
                "property_code": property_code,
                "url": url,
                "page_title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
        print(f"   {property_code:8s} {url}  ->  {len(chunks)} chunks")
        time.sleep(0.5)  # be polite

    if all_ids:
        coll.add(ids=all_ids, documents=all_docs, metadatas=all_meta)
    return len(all_ids)


def ingest_all() -> None:
    coll, backend = get_collection()
    print(f"Embedding backend: {backend}")
    print(f"Chroma dir:        {_settings.chroma_dir}")
    print()

    total = 0
    for code, urls in SCRAPE_SEEDS.items():
        print(f"-- {code} -------------------------------------")
        total += index_property(coll, code, urls)

    print()
    print(f"Total chunks indexed: {total}")
    print(f"Collection size:      {coll.count()}")


if __name__ == "__main__":
    ingest_all()
