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

from ..config import get_settings
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


# ---------------------------------------------------------------------------
# v2 — docling + Jina v4 multimodal
# ---------------------------------------------------------------------------

# CLIP image-vs-text cosine distances sit in a different band than
# text-vs-text (typically 0.8-1.1 vs 0.4-0.7). Pure-distance filtering picks
# up irrelevant lifestyle photos for abstract queries like "amenities" — so
# we ALSO boost candidates whose source URL or section_path matches a
# page-section keyword in the user's query (e.g. "gym pool" → /amenities/).
_DISTANCE_THRESHOLD_IMAGE = 1.05
_MAX_IMAGES = 3
_IMAGE_QUERY_K = 12  # over-fetch so the boost has room to re-rank

# Map query keywords → URL substrings that suggest a relevant page section.
# When the query contains any of the keys, images whose URL or section_path
# contains the corresponding value get a 0.25 distance bonus (lower = better).
_SECTION_HINTS = {
    "amenit":      ["amenit"],
    "gallery":     ["gallery", "gallerie"],
    "photo":       ["gallery", "gallerie"],
    "picture":     ["gallery", "gallerie"],
    "image":       ["gallery", "gallerie"],
    "floor plan":  ["floorplan", "floor-plan"],
    "floorplan":   ["floorplan", "floor-plan"],
    "neighborhood":["neighborhood", "neighbourhood"],
    "neighbor":    ["neighborhood", "neighbourhood"],
    "amenities":   ["amenit"],
    "pool":        ["amenit", "gallery"],
    "gym":         ["amenit"],
    "fitness":     ["amenit"],
    "kitchen":     ["amenit", "gallery", "feature"],
    "bedroom":     ["floorplan", "gallery"],
    "lounge":      ["amenit", "gallery"],
    "clubroom":    ["amenit"],
    "lobby":       ["amenit", "gallery"],
}
_BOOST_DISTANCE = 0.25  # subtracted from candidate distance on hint match


def _hint_terms_for(query: str) -> list[str]:
    """Return URL substrings to boost based on keywords in `query`."""
    q = (query or "").lower()
    hits: list[str] = []
    for trigger, url_hints in _SECTION_HINTS.items():
        if trigger in q:
            hits.extend(url_hints)
    # Dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def search_property_v2(
    property_code: str,
    query: str,
    k: int = 8,
) -> dict[str, Any]:
    """Query the v2 collection. Returns text chunks + a separate `images` list.

    The returned `images` carry public `/doc_store/...` URLs ready to be
    rendered as an `image` UIComponent by the chat composer.
    """
    code = require_scope(property_code)
    if not query or not query.strip():
        return {
            "property_code": code, "query": query, "row_count": 0,
            "chunks": [], "images": [], "sources": [],
        }
    k = max(1, min(int(k or DEFAULT_K), MAX_K))

    from ..ingestion.v2.doc_store import public_url
    from ..ingestion.v2.embedder import JinaV4Embedder
    from ..ingestion.v2.pipeline import get_collection_v2

    coll = get_collection_v2()
    embedder = JinaV4Embedder.get()
    qvec = embedder.embed_query(query)

    # Pass 1: text/table chunks for the LLM context.
    res = coll.query(
        query_embeddings=[qvec],
        n_results=k,
        where={
            "$and": [
                {"property_code": code},
                {"modality": {"$in": ["text", "table"]}},
            ]
        },
    )
    docs  = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    chunks: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if (meta or {}).get("property_code") != code:
            continue
        chunks.append({
            "text": doc,
            "url": meta.get("url"),
            "page_title": meta.get("page_title"),
            "section_path": meta.get("section_path"),
            "modality": meta.get("modality", "text"),
            "chunk_index": meta.get("chunk_index"),
            "distance": float(dist) if dist is not None else None,
            "caption": meta.get("caption"),
            "image_path": meta.get("image_path"),
            "table_html": meta.get("table_html"),
        })

    # Pass 2: image chunks scored against the same query vector. CLIP image
    # distances sit in a different band than text-text, so we keep them in
    # their own ranking and surface the top few.
    images: list[dict[str, Any]] = []
    seen_image_paths: set[str] = set()
    try:
        img_res = coll.query(
            query_embeddings=[qvec],
            n_results=_IMAGE_QUERY_K,
            where={
                "$and": [
                    {"property_code": code},
                    {"modality": "image"},
                ]
            },
        )
    except Exception:  # noqa: BLE001
        img_res = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    img_metas = (img_res.get("metadatas") or [[]])[0]
    img_dists = (img_res.get("distances") or [[]])[0]

    hint_terms = _hint_terms_for(query)

    # Re-rank: apply URL-keyword boost so a "show me amenities" query
    # prefers images scraped from /amenities/ over lifestyle gallery photos.
    candidates: list[dict[str, Any]] = []
    for meta, dist in zip(img_metas, img_dists):
        if (meta or {}).get("property_code") != code:
            continue
        path = meta.get("image_path")
        if not path:
            continue
        d = float(dist) if dist is not None else 1.0
        url = (meta.get("url") or "").lower()
        section = (meta.get("section_path") or "").lower()
        caption = (meta.get("caption") or "").lower()
        # Boost if any hint term appears in URL/section/caption.
        matched_hint = False
        if hint_terms:
            for h in hint_terms:
                if h in url or h in section or h in caption:
                    matched_hint = True
                    break
        adjusted = d - _BOOST_DISTANCE if matched_hint else d
        candidates.append({
            "path": path, "meta": meta, "raw_dist": d,
            "adj_dist": adjusted, "boosted": matched_hint,
        })

    # Sort by adjusted distance ascending (closer = better).
    candidates.sort(key=lambda c: c["adj_dist"])

    for c in candidates:
        path = c["path"]
        if path in seen_image_paths:
            continue
        # When the query has section hints but no images matched them,
        # apply a stricter raw-distance cap so we don't surface irrelevant
        # lifestyle photos. Otherwise use the normal threshold.
        cap = _DISTANCE_THRESHOLD_IMAGE
        if hint_terms and not c["boosted"]:
            cap = 0.92
        if c["raw_dist"] > cap:
            continue
        meta = c["meta"]
        seen_image_paths.add(path)
        images.append({
            "url": public_url(path),
            "caption": meta.get("caption") or meta.get("section_path") or "",
            "source_url": meta.get("url"),
            "section_path": meta.get("section_path"),
            "distance": c["raw_dist"],
            "boosted": c["boosted"],
        })
        if len(images) >= _MAX_IMAGES:
            break

    # Dedupe sources by page URL while preserving order.
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for ch in chunks:
        u = ch["url"]
        if u and u not in seen:
            seen.add(u)
            label = ch.get("page_title") or u
            if "|" in label:
                label = label.split("|", 1)[0].strip()
            sources.append({"label": label, "url": u})

    return {
        "property_code": code,
        "query": query,
        "row_count": len(chunks),
        "chunks": chunks,
        "images": images,
        "sources": sources,
    }


def build_context_block_v2(chunks: list[dict[str, Any]]) -> str:
    """Like build_context_block but flags table/image chunks so the LLM knows."""
    if not chunks:
        return ""
    parts = []
    for i, ch in enumerate(chunks, 1):
        url = ch.get("url") or ""
        title = ch.get("page_title") or ""
        sect = ch.get("section_path") or ""
        modality = ch.get("modality", "text")
        body = (ch.get("text") or "").strip()
        tag = f"Source {i} — {title}"
        if sect:
            tag += f" — {sect}"
        if modality == "image":
            cap = ch.get("caption") or ""
            parts.append(f"[{tag} (IMAGE)]\n{cap}\n(URL: {url})")
        elif modality == "table":
            parts.append(f"[{tag} (TABLE)]\n{body}\n(URL: {url})")
        else:
            parts.append(f"[{tag}]\n{body}\n(URL: {url})")
    return "\n\n".join(parts)


def search_property_active(property_code: str, query: str, k: int = DEFAULT_K) -> dict[str, Any]:
    """Dispatch to v1 or v2 based on RAG_VERSION setting."""
    version = (get_settings().rag_version or "v2").lower()
    if version == "v1":
        return search_property(property_code, query, k=k)
    return search_property_v2(property_code, query, k=k)


def has_content(property_code: str) -> bool:
    """Cheap check — does this property have any indexed chunks?"""
    code = require_scope(property_code)
    coll = _get_collection()
    try:
        res = coll.get(where={"property_code": code}, limit=1)
        return bool(res.get("ids"))
    except Exception:
        return False
