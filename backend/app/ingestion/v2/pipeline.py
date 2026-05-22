"""End-to-end ingestion pipeline for RAG v2.

Per URL:
  1. docling extract → ordered Blocks
  2. structure-aware chunker → ordered Chunks (text / table / image)
  3. images & tables saved to the local doc store
  4. text + image chunks embedded via Jina v4 (modality-aware)
  5. upsert into Chroma `property_chunks_v2`

Property scope is the only `where` filter the retrieval side uses, so each
chunk carries `property_code` in its metadata.
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

_collection = None
_collection_lock = threading.Lock()


def _safe_url_slug(url: str) -> str:
    path = urlparse(url).path or "root"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_") or "root"
    return slug[:80]


def _stable_id(property_code: str, url: str, chunk_index: int, modality: str) -> str:
    return f"{property_code}__{_safe_url_slug(url)}__{modality}__{chunk_index:03d}"


def get_collection_v2():
    """Open (or create) the v2 collection. Embeddings supplied at write/read time."""
    global _collection
    if _collection is not None:
        return _collection
    with _collection_lock:
        if _collection is not None:
            return _collection
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        settings = get_settings()
        Path(settings.chroma_dir_v2).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=settings.chroma_dir_v2,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # No embedding_function — we precompute on write and pass
        # `query_embeddings` on read so text & image vectors share a space.
        _collection = client.get_or_create_collection(
            name=settings.collection_v2,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _meta_for_chroma(d: dict[str, Any]) -> dict[str, Any]:
    """Chroma only stores scalar metadata. Drop None and stringify the rest."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def ingest_urls(
    property_code: str,
    urls: list[str],
    replace: bool = False,
) -> dict[str, Any]:
    """Index a list of URLs into the v2 collection for one property."""
    coll = get_collection_v2()
    embedder = JinaV4Embedder.get()

    if replace:
        try:
            coll.delete(where={"property_code": property_code})
        except Exception as e:  # noqa: BLE001
            log.warning("delete-by-property failed (ok on empty collection): %s", e)

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
        # Persist image bytes; replace metadata with public path.
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

    # Embed in two passes: text-like (text + table) and image (the image
    # chunks). Tables are embedded as their markdown text — fine because the
    # retrieval-side query is always text.
    text_chunks = [c for c in all_chunks if c.modality in ("text", "table")]
    image_chunks = [c for c in all_chunks if c.modality == "image"]

    embeddings_by_idx: dict[int, list[float]] = {}
    if text_chunks:
        text_vecs = embedder.embed_text([c.embed_input for c in text_chunks])
        for c, v in zip(text_chunks, text_vecs):
            embeddings_by_idx[id(c)] = v

    if image_chunks:
        # Some platforms can't load the model in image mode (CPU-only without
        # vision deps). Fall back to embedding the caption text instead so the
        # image chunk is still searchable.
        try:
            # We don't have raw bytes anymore (already saved); reload from disk.
            settings = get_settings()
            root = Path(settings.doc_store_dir)
            paths = [(root / c.metadata["image_path"]).read_bytes() for c in image_chunks]
            img_vecs = embedder.embed_image(paths)
        except Exception as e:  # noqa: BLE001
            log.warning("image embedding failed (%s); falling back to caption text", e)
            img_vecs = embedder.embed_text([c.embed_input or "image" for c in image_chunks])
        for c, v in zip(image_chunks, img_vecs):
            embeddings_by_idx[id(c)] = v

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    embs: list[list[float]] = []
    for ch in all_chunks:
        cid = _stable_id(
            property_code,
            ch.metadata.get("url", ""),
            int(ch.metadata.get("chunk_index", 0)),
            ch.modality,
        )
        ids.append(cid)
        docs.append(ch.embed_input)
        metas.append(_meta_for_chroma(ch.metadata))
        embs.append(embeddings_by_idx[id(ch)])

    # Upsert in modest batches.
    BATCH = 64
    for i in range(0, len(ids), BATCH):
        coll.upsert(
            ids=ids[i:i + BATCH],
            documents=docs[i:i + BATCH],
            metadatas=metas[i:i + BATCH],
            embeddings=embs[i:i + BATCH],
        )

    summary["chunks_indexed"] = len(ids)
    return summary
