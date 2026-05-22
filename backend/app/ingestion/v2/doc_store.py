"""Local filesystem doc store for image + table artifacts.

Layout: {DOC_STORE_DIR}/{property_code}/{sha256}.{ext}

Designed to be S3-swappable later — `save_artifact` returns a relative path
that the FastAPI app mounts statically at `/doc_store/...` and the frontend
loads as `<img src="/doc_store/...">`.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ...config import get_settings


_SAFE_CODE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_property(property_code: str) -> str:
    cleaned = _SAFE_CODE.sub("_", property_code or "").strip("_")
    return cleaned or "_unknown"


def _sniff_ext(image_bytes: bytes, default: str) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "jpg"
    if image_bytes[:4] == b"GIF8":
        return "gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes[:4] == b"<svg" or image_bytes[:5] == b"<?xml":
        return "svg"
    return (default or "bin").lstrip(".").lower()


def save_artifact(property_code: str, data: bytes, ext: str = "bin") -> str:
    """Write `data` under the property's doc-store dir; return the relative path.

    Idempotent: if the same content already exists, the existing file is
    reused. The returned path is relative to the doc-store root, e.g.
    `115r/ab12cd34...png`. Prepend `/doc_store/` for HTTP URLs.
    """
    settings = get_settings()
    root = Path(settings.doc_store_dir)
    sub = _safe_property(property_code)
    target_dir = root / sub
    target_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(data).hexdigest()
    resolved_ext = _sniff_ext(data, ext)
    rel = f"{sub}/{digest}.{resolved_ext}"
    full = root / rel
    if not full.exists():
        full.write_bytes(data)
    return rel


def public_url(rel_path: str) -> str:
    """Convert a stored relative path into the URL the frontend uses."""
    return f"/doc_store/{rel_path.lstrip('/')}"
