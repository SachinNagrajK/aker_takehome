"""RAG retrieval tool — Chroma collection wrapped with property scoping.

This is the only place outside ingestion that touches the vector store.
The contract mirrors `sql_tools.py`:

  1. First line of every public function is `require_scope(property_code)`.
  2. Every Chroma query passes `where={"property_code": code}` — the metadata
     filter is the hard guarantee at the vector-store layer.
  3. Returns plain dicts ready for the response composer / LLM context.

The collection itself is created and populated by
`app/ingestion/scrape_and_index.py`. Here we only read from it.
"""
from __future__ import annotations

import threading
from typing import Any

from ..guardrails.scope import require_scope


# ---------------------------------------------------------------------------
# Collection handle (lazy + thread-safe singleton)
# ---------------------------------------------------------------------------

_collection = None
_collection_lock = threading.Lock()


def _get_collection():
    """Return the shared Chroma collection, opening it once."""
    global _collection
    if _collection is not None:
        return _collection
    with _collection_lock:
        if _collection is None:
            # Reuse the same factory used by the indexer so embedding
            # functions stay aligned between write and read paths.
            from ..ingestion.scrape_and_index import get_collection
            _collection, _ = get_collection()
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_K = 4
MAX_K = 10


def search_property(
    property_code: str,
    query: str,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Top-k chunks for `query`, hard-filtered to `property_code`.

    Returns:
        {
          "property_code": str,
          "query": str,
          "row_count": int,
          "chunks": [
            {
              "text": str,
              "url": str,
              "page_title": str,
              "chunk_index": int,
              "distance": float,
            }, ...
          ],
          "sources": [{"label": str, "url": str}, ...],   # deduped by URL
        }
    """
    code = require_scope(property_code)
    if not query or not query.strip():
        return {
            "property_code": code, "query": query, "row_count": 0,
            "chunks": [], "sources": [],
        }
    k = max(1, min(int(k or DEFAULT_K), MAX_K))

    coll = _get_collection()

    # The `where` filter is the scope guarantee at the vector layer.
    res = coll.query(
        query_texts=[query],
        n_results=k,
        where={"property_code": code},
    )

    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # Defensive: even if Chroma somehow returned a row from another
        # property, drop it. (Should never happen with `where` filter.)
        if meta.get("property_code") != code:
            continue
        chunks.append({
            "text": doc,
            "url": meta.get("url"),
            "page_title": meta.get("page_title"),
            "chunk_index": meta.get("chunk_index"),
            "distance": float(dist) if dist is not None else None,
        })

    # Dedupe sources by URL while preserving order.
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for ch in chunks:
        u = ch["url"]
        if u and u not in seen:
            seen.add(u)
            label = ch.get("page_title") or u
            # Trim a typical "X | Property Name" suffix in page titles.
            if "|" in label:
                label = label.split("|", 1)[0].strip()
            sources.append({"label": label, "url": u})

    return {
        "property_code": code,
        "query": query,
        "row_count": len(chunks),
        "chunks": chunks,
        "sources": sources,
    }


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Concat top-k chunks into a single context string for the LLM prompt."""
    if not chunks:
        return ""
    parts = []
    for i, ch in enumerate(chunks, 1):
        url = ch.get("url") or ""
        title = ch.get("page_title") or ""
        body = (ch.get("text") or "").strip()
        parts.append(f"[Source {i} — {title}]\n{body}\n(URL: {url})")
    return "\n\n".join(parts)


def has_content(property_code: str) -> bool:
    """Cheap check — does this property have any indexed chunks?"""
    code = require_scope(property_code)
    coll = _get_collection()
    try:
        res = coll.get(where={"property_code": code}, limit=1)
        return bool(res.get("ids"))
    except Exception:
        return False
