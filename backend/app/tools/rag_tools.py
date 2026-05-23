"""RAG retrieval tool — Pinecone serverless wrapped with property scoping.

This is the only place outside ingestion that touches the vector store.
The contract mirrors `sql_tools.py`:

  1. First line of every public function is `require_scope(property_code)`.
  2. Every Pinecone query uses `namespace=code` — namespace-per-property is
     the hard scope guarantee at the vector-store layer (no metadata filter
     can leak across properties).
  3. Returns plain dicts ready for the response composer / LLM context.

The index itself is created and populated by
`app/ingestion/v2/pipeline.py`. Here we only read from it.
"""
from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..guardrails.scope import require_scope


# Pinecone returns cosine SIMILARITY in `score` (1 = identical, -1 = opposite).
# The legacy code assumed cosine DISTANCE (0 = identical, 2 = opposite), so
# convert at the boundary and keep all downstream thresholds unchanged.
def _to_distance(score: float | None) -> float:
    if score is None:
        return 1.0
    return float(1.0 - score)


# ---------------------------------------------------------------------------
# Public API — v1 (text-only). Retained as a thin shim around v2 so the
# graph can still call search_property_active() without a flag check on
# the v1 path. The actual retrieval is v2-flavoured (Pinecone).
# ---------------------------------------------------------------------------

DEFAULT_K = 4
MAX_K = 10


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
# v2 — docling + Jina v4 multimodal, on Pinecone
# ---------------------------------------------------------------------------

# CLIP image-vs-text cosine distances sit in a different band than
# text-vs-text (typically 0.8-1.1 vs 0.4-0.7). Pure-distance filtering picks
# up irrelevant lifestyle photos for abstract queries like "amenities" — so
# we ALSO boost candidates whose source URL or section_path matches a
# page-section keyword in the user's query (e.g. "gym pool" → /amenities/).
_DISTANCE_THRESHOLD_IMAGE = 1.05
_DEFAULT_MAX_IMAGES = 3
_HARD_CAP_MAX_IMAGES = 25       # absolute ceiling when caller asks for "all"
_IMAGE_QUERY_K_MULT  = 4        # over-fetch this many * max so boost has room

# Map query keywords → URL substrings that suggest a relevant page section.
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
    max_images: int = _DEFAULT_MAX_IMAGES,
) -> dict[str, Any]:
    """Query the v2 Pinecone index. Returns text chunks + a separate `images` list.

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
    max_images = max(1, min(int(max_images or _DEFAULT_MAX_IMAGES), _HARD_CAP_MAX_IMAGES))
    image_query_k = min(max_images * _IMAGE_QUERY_K_MULT, 60)

    from ..ingestion.v2.embedder import JinaV4Embedder
    from ..ingestion.v2.pipeline import get_index_v2

    index = get_index_v2()
    embedder = JinaV4Embedder.get()
    qvec = embedder.embed_query(query)

    # Pass 1: text/table chunks for the LLM context.
    res = index.query(
        vector=qvec,
        top_k=k,
        namespace=code,
        filter={"modality": {"$in": ["text", "table"]}},
        include_metadata=True,
    )
    chunks: list[dict[str, Any]] = []
    for m in (res.get("matches") or []):
        meta = m.get("metadata") or {}
        chunks.append({
            "text": meta.get("text"),
            "url": meta.get("url"),
            "page_title": meta.get("page_title"),
            "section_path": meta.get("section_path"),
            "modality": meta.get("modality", "text"),
            "chunk_index": meta.get("chunk_index"),
            "distance": _to_distance(m.get("score")),
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
        img_res = index.query(
            vector=qvec,
            top_k=image_query_k,
            namespace=code,
            filter={"modality": "image"},
            include_metadata=True,
        )
    except Exception:  # noqa: BLE001
        img_res = {"matches": []}

    hint_terms = _hint_terms_for(query)

    candidates: list[dict[str, Any]] = []
    for m in (img_res.get("matches") or []):
        meta = m.get("metadata") or {}
        path = meta.get("image_path")
        if not path:
            continue
        d = _to_distance(m.get("score"))
        url = (meta.get("url") or "").lower()
        section = (meta.get("section_path") or "").lower()
        caption = (meta.get("caption") or "").lower()
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

    candidates.sort(key=lambda c: c["adj_dist"])

    from ..ingestion.v2.doc_store import public_url
    for c in candidates:
        path = c["path"]
        if path in seen_image_paths:
            continue
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
        if len(images) >= max_images:
            break

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


def search_property_active(
    property_code: str,
    query: str,
    k: int = DEFAULT_K,
    max_images: int = _DEFAULT_MAX_IMAGES,
) -> dict[str, Any]:
    """Dispatch to v2 retrieval. (v1 path retired with the Chroma swap.)"""
    return search_property_v2(property_code, query, k=k, max_images=max_images)


def has_content(property_code: str) -> bool:
    """Cheap check — does this property have any indexed chunks?

    Uses Pinecone's per-namespace stats so we don't pay for a query.
    """
    code = require_scope(property_code)
    try:
        from ..ingestion.v2.pipeline import get_index_v2
        index = get_index_v2()
        stats = index.describe_index_stats()
        ns = (stats.get("namespaces") or {}).get(code)
        if not ns:
            return False
        return int(ns.get("vector_count", 0)) > 0
    except Exception:
        return False
