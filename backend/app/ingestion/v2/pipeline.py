"""End-to-end ingestion pipeline for RAG v2.

Per URL:
  1. docling extract → ordered Blocks
  2. structure-aware chunker → ordered Chunks (text / table / image)
  3. images & tables saved to the local doc store
  4. text + image chunks embedded via Jina v4 (modality-aware)
  5. upsert into Pinecone serverless index, namespace = property_code

Each property gets its own Pinecone namespace, so retrieval doesn't need a
property_code metadata filter — the namespace is the scope guarantee.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ...config import get_settings
from .chunker import Chunk, chunk_blocks
from .doc_store import save_artifact
from .embedder import JinaV4Embedder
from .extractor import extract


log = logging.getLogger(__name__)

# Jina-CLIP-v2 produces 1024-dim vectors.
_EMBED_DIM = 1024

# Pinecone metadata is capped at 40 KB per vector — give text chunks plenty
# of headroom but truncate aggressively if a single chunk is enormous.
_MAX_TEXT_BYTES = 30_000

_index = None
_index_lock = threading.Lock()


def _safe_url_slug(url: str) -> str:
    path = urlparse(url).path or "root"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") or "root"
    return slug[:80]


def _stable_id(property_code: str, url: str, chunk_index: int, modality: str) -> str:
    return f"{property_code}__{_safe_url_slug(url)}__{modality}__{chunk_index:03d}"


def get_index_v2():
    """Open (or create) the Pinecone serverless index. Lazy singleton."""
    global _index
    if _index is not None:
        return _index
    with _index_lock:
        if _index is not None:
            return _index
        from pinecone import Pinecone, ServerlessSpec

        s = get_settings()
        if not s.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set — add it to backend/.env"
            )
        pc = Pinecone(api_key=s.pinecone_api_key)

        existing = {ix["name"] for ix in pc.list_indexes()}
        if s.pinecone_index not in existing:
            log.info("Creating Pinecone index %s (dim=%d, cosine, %s/%s)",
                     s.pinecone_index, _EMBED_DIM, s.pinecone_cloud, s.pinecone_region)
            pc.create_index(
                name=s.pinecone_index,
                dimension=_EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
        _index = pc.Index(s.pinecone_index)
    return _index


def _meta_for_pinecone(d: dict[str, Any], text: str) -> dict[str, Any]:
    """Pinecone metadata: scalars (str/int/float/bool) and list[str] only.
    Drop None, stringify everything else, and store the chunk text under
    `text` so retrieval can return it without a separate document store.
    """
    out: dict[str, Any] = {}
    if text:
        body = text if len(text.encode("utf-8")) <= _MAX_TEXT_BYTES else text[: _MAX_TEXT_BYTES // 4]
        out["text"] = body
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def ingest_urls(
    property_code: str,
    urls: list[str],
    replace: bool = False,
) -> dict[str, Any]:
    """Index a list of URLs into the Pinecone namespace for one property."""
    index = get_index_v2()
    embedder = JinaV4Embedder.get()

    if replace:
        try:
            index.delete(delete_all=True, namespace=property_code)
        except Exception as e:  # noqa: BLE001
            log.warning("delete-all in namespace failed (ok if empty): %s", e)

    summary = {
        "property_code": property_code,
        "urls_processed": 0,
        "chunks_indexed": 0,
        "images_stored": 0,
        "tables_stored": 0,
        "errors": [],
    }

    all_chunks: list[Chunk] = []

    for url in urls:
        try:
            blocks = extract(url)
        except Exception as e:  # noqa: BLE001
            log.exception("docling extract failed for %s", url)
            summary["errors"].append({"url": url, "stage": "extract", "error": str(e)})
            continue
        chunks = chunk_blocks(blocks)
        if not chunks:
            summary["errors"].append({"url": url, "stage": "chunk", "error": "no chunks produced"})
            continue
        for ch in chunks:
            ch.metadata["property_code"] = property_code
            ch.metadata["modality"] = ch.modality
            if ch.modality == "image" and ch.image_bytes:
                rel = save_artifact(property_code, ch.image_bytes, ch.image_ext or "png")
                ch.metadata["image_path"] = rel  # frontend builds `/doc_store/{rel}`
                ch.image_bytes = None
                summary["images_stored"] += 1
            elif ch.modality == "table":
                summary["tables_stored"] += 1
        all_chunks.extend(chunks)
        summary["urls_processed"] += 1

    if not all_chunks:
        return summary

    # Embed in two passes: text-like (text + table) and image. Tables are
    # embedded as their markdown text — fine because retrieval always queries
    # with a text vector.
    text_chunks = [c for c in all_chunks if c.modality in ("text", "table")]
    image_chunks = [c for c in all_chunks if c.modality == "image"]

    embeddings_by_idx: dict[int, list[float]] = {}
    if text_chunks:
        text_vecs = embedder.embed_text([c.embed_input for c in text_chunks])
        for c, v in zip(text_chunks, text_vecs):
            embeddings_by_idx[id(c)] = v

    if image_chunks:
        try:
            settings = get_settings()
            root = Path(settings.doc_store_dir)
            paths = [(root / c.metadata["image_path"]).read_bytes() for c in image_chunks]
            img_vecs = embedder.embed_image(paths)
        except Exception as e:  # noqa: BLE001
            log.warning("image embedding failed (%s); falling back to caption text", e)
            img_vecs = embedder.embed_text([c.embed_input or "image" for c in image_chunks])
        for c, v in zip(image_chunks, img_vecs):
            embeddings_by_idx[id(c)] = v

    vectors: list[dict[str, Any]] = []
    dropped_zero = 0
    for ch in all_chunks:
        vec = embeddings_by_idx[id(ch)]
        # Pinecone rejects all-zero vectors (cosine undefined). The embedder
        # uses zeros as a sentinel for images PIL couldn't decode; skip those.
        if not any(vec):
            dropped_zero += 1
            continue
        cid = _stable_id(
            property_code,
            ch.metadata.get("url", ""),
            int(ch.metadata.get("chunk_index", 0)),
            ch.modality,
        )
        vectors.append({
            "id": cid,
            "values": vec,
            "metadata": _meta_for_pinecone(ch.metadata, ch.embed_input),
        })
    if dropped_zero:
        log.warning("dropped %d chunks with zero-vector embedding (unreadable images)", dropped_zero)
        summary.setdefault("dropped_unreadable", 0)
        summary["dropped_unreadable"] = dropped_zero

    # Pinecone recommends batches of <= 100 vectors per upsert.
    BATCH = 96
    for i in range(0, len(vectors), BATCH):
        index.upsert(vectors=vectors[i:i + BATCH], namespace=property_code)

    summary["chunks_indexed"] = len(vectors)
    return summary
