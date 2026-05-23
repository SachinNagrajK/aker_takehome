"""Doc store for image + table artifacts — backed by Supabase Storage.

Layout (key inside the bucket): {property_code}/{sha256}.{ext}

`save_artifact()` uploads the bytes to the configured Supabase Storage
bucket and returns the *key* (relative path, e.g. `115r/ab12cd34...png`).
The key is what we store in Pinecone metadata as `image_path`.

`public_url(key)` converts the stored key into a fully-qualified URL the
frontend can drop straight into `<img src=…>`. For the public Supabase
bucket the format is:
  https://<project>.supabase.co/storage/v1/object/public/<bucket>/<key>

This design replaces the previous local-filesystem doc store served by
FastAPI's StaticFiles mount, so the deployed backend can stay stateless
on Hugging Face Spaces (whose container disk is ephemeral).
"""
from __future__ import annotations

import hashlib
import logging
import re

import requests

from ...config import get_settings


log = logging.getLogger(__name__)

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


_EXT_TO_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "bin": "application/octet-stream",
}


def _storage_endpoint(key: str, public: bool = False) -> str:
    s = get_settings()
    base = (s.supabase_url or "").rstrip("/")
    bucket = s.supabase_storage_bucket
    if public:
        return f"{base}/storage/v1/object/public/{bucket}/{key}"
    return f"{base}/storage/v1/object/{bucket}/{key}"


def save_artifact(property_code: str, data: bytes, ext: str = "bin") -> str:
    """Upload `data` to Supabase Storage; return the bucket key.

    Idempotent on content: identical bytes hash to the same key and are
    upserted (no duplicate storage). The returned key is what's saved in
    Pinecone metadata as `image_path`.
    """
    s = get_settings()
    if not s.supabase_url or not s.supabase_service_role_key:
        raise RuntimeError(
            "Supabase Storage not configured — set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in backend/.env"
        )

    sub = _safe_property(property_code)
    digest = hashlib.sha256(data).hexdigest()
    resolved_ext = _sniff_ext(data, ext)
    key = f"{sub}/{digest}.{resolved_ext}"

    headers = {
        "Authorization": f"Bearer {s.supabase_service_role_key}",
        "Content-Type": _EXT_TO_MIME.get(resolved_ext, "application/octet-stream"),
        "x-upsert": "true",  # idempotent — overwrite if same key exists
        "cache-control": "public, max-age=31536000, immutable",
    }
    url = _storage_endpoint(key, public=False)
    r = requests.post(url, headers=headers, data=data, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(
            f"Supabase Storage upload failed ({r.status_code}): {r.text[:300]}"
        )
    return key


def public_url(key: str) -> str:
    """Convert a stored bucket key into a fully-qualified public URL."""
    return _storage_endpoint(key.lstrip("/"), public=True)
