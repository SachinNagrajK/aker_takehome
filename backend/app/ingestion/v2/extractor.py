"""Docling-based structured extractor.

Returns an ordered list of Block dicts so the chunker can preserve document
structure (headings → section paths, tables/images as their own chunks).

We stay defensive about docling's evolving API — the 2.91 release exposes
`DocumentConverter` and a `DoclingDocument` with `texts`, `tables`,
`pictures`, `iterate_items()`. Image binary access varies by source so we
try a few paths.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests

log = logging.getLogger(__name__)


@dataclass
class Block:
    type: str                       # "heading" | "text" | "table" | "image"
    content: str = ""               # text body / markdown / table-markdown / image caption
    level: int | None = None        # heading depth (1..6) when type == "heading"
    html_table: str | None = None
    image_bytes: bytes | None = None
    image_ext: str | None = None
    source_url: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


def _fetch(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, headers=_REQ_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def _picture_to_bytes(pic, doc) -> tuple[bytes | None, str]:
    """Best-effort: extract image bytes + extension from a docling Picture."""
    try:
        pil = pic.get_image(doc)
        if pil is not None:
            buf = io.BytesIO()
            fmt = (pil.format or "PNG").upper()
            ext = "png" if fmt == "PNG" else ("jpg" if fmt in ("JPEG", "JPG") else fmt.lower())
            pil.save(buf, format="PNG" if ext == "png" else fmt)
            return buf.getvalue(), ext
    except Exception as e:  # noqa: BLE001
        log.debug("picture.get_image failed: %s", e)

    # Fallbacks for older docling shapes
    for attr in ("image", "_image"):
        obj = getattr(pic, attr, None)
        if obj is None:
            continue
        for byte_attr in ("bytes", "data", "raw"):
            b = getattr(obj, byte_attr, None)
            if isinstance(b, (bytes, bytearray)):
                return bytes(b), getattr(obj, "format", "png") or "png"

    uri = getattr(pic, "uri", None) or getattr(getattr(pic, "image", None), "uri", None)
    if uri and isinstance(uri, str) and uri.startswith("http"):
        try:
            return _fetch(uri), uri.rsplit(".", 1)[-1][:5] or "png"
        except Exception as e:  # noqa: BLE001
            log.debug("picture uri fetch failed: %s", e)

    return None, "png"


def _table_to_markdown_and_html(tbl, doc) -> tuple[str, str]:
    md, html = "", ""
    try:
        md = tbl.export_to_markdown(doc=doc)
    except TypeError:
        try:
            md = tbl.export_to_markdown()
        except Exception:  # noqa: BLE001
            md = ""
    except Exception:  # noqa: BLE001
        md = ""
    try:
        html = tbl.export_to_html(doc=doc)
    except TypeError:
        try:
            html = tbl.export_to_html()
        except Exception:  # noqa: BLE001
            html = ""
    except Exception:  # noqa: BLE001
        html = ""
    return md, html


def _heading_level(item) -> int | None:
    """Map docling heading labels to depth (1..6). Returns None for non-headings."""
    label = getattr(item, "label", None)
    label_str = str(label).lower() if label is not None else ""
    if "title" in label_str:
        return 1
    if "section_header" in label_str or "header" in label_str or "heading" in label_str:
        return getattr(item, "level", None) or 2
    return None


def extract(url: str) -> list[Block]:
    """Convert a URL via docling and return ordered Blocks.

    Only blocks with non-empty content (or image bytes) are returned.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(url)
    doc = result.document

    blocks: list[Block] = []

    # Prefer iterate_items — preserves document order across headings/text/tables/pictures.
    iterator: Iterable
    try:
        iterator = (item for item, _level in doc.iterate_items())
    except Exception:  # noqa: BLE001
        # Fallback: stitch the typed collections together (loses interleaving).
        iterator = list(getattr(doc, "texts", []) or []) + \
                   list(getattr(doc, "tables", []) or []) + \
                   list(getattr(doc, "pictures", []) or [])

    seen_ids: set[int] = set()
    for item in iterator:
        if id(item) in seen_ids:
            continue
        seen_ids.add(id(item))

        cls = type(item).__name__

        if cls.endswith("PictureItem") or cls == "Picture":
            img_bytes, ext = _picture_to_bytes(item, doc)
            caption = ""
            try:
                cap = item.caption_text(doc=doc)
                if cap:
                    caption = cap.strip()
            except Exception:  # noqa: BLE001
                caption = ""
            if img_bytes:
                blocks.append(Block(
                    type="image",
                    content=caption,
                    image_bytes=img_bytes,
                    image_ext=ext,
                    source_url=url,
                ))
            continue

        if cls.endswith("TableItem") or cls == "Table":
            md, html = _table_to_markdown_and_html(item, doc)
            if md or html:
                blocks.append(Block(
                    type="table",
                    content=md or "",
                    html_table=html or None,
                    source_url=url,
                ))
            continue

        # Text-like item (paragraph, heading, list item, code, etc.)
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        level = _heading_level(item)
        if level is not None:
            blocks.append(Block(type="heading", content=text, level=level, source_url=url))
        else:
            blocks.append(Block(type="text", content=text, source_url=url))

    page_title = ""
    try:
        page_title = (doc.name or "").strip()
    except Exception:  # noqa: BLE001
        page_title = ""
    if page_title:
        # Stash title on the first block so the chunker can hoist it into metadata.
        blocks.insert(0, Block(type="heading", content=page_title, level=1, source_url=url,
                               extra={"is_page_title": True}))

    # docling 2.x HTML backend ignores <img> tags. For marketing sites where
    # images ARE the content (gallery, floor plans, amenity photos), fall back
    # to a BeautifulSoup pass that scrapes <img> + lazy-load attrs and appends
    # image Blocks. Headings/text already came from docling above.
    blocks.extend(_scrape_html_images(url))
    return blocks


# ---------------------------------------------------------------------------
# BeautifulSoup image fallback
# ---------------------------------------------------------------------------

_IMG_MIN_BYTES = 8 * 1024     # skip tracking pixels / tiny icons
_IMG_MAX_BYTES = 6 * 1024 * 1024
_IMG_MAX_PER_PAGE = 25
_SKIP_EXTS = {"svg", "ico"}
_SKIP_HINTS = ("logo", "icon", "favicon", "sprite", "pixel", "tracking")


def _candidate_src(img_tag) -> str | None:
    """Return the best image URL from an <img> tag (handles lazy-load attrs)."""
    for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-srcset"):
        v = img_tag.get(attr)
        if v:
            # srcset attrs may carry "url 1x, url 2x"; take the last (largest).
            if "," in v and attr.endswith("srcset"):
                v = v.split(",")[-1].strip().split(" ")[0]
            return v
    # As a last resort, parse srcset on the tag.
    srcset = img_tag.get("srcset")
    if srcset:
        return srcset.split(",")[-1].strip().split(" ")[0]
    return None


def _nearest_caption(img_tag) -> str:
    alt = (img_tag.get("alt") or "").strip()
    if alt:
        return alt
    title = (img_tag.get("title") or "").strip()
    if title:
        return title
    # Walk up to a <figure> with a <figcaption>.
    parent = img_tag.parent
    for _ in range(4):
        if parent is None:
            break
        if parent.name == "figure":
            cap = parent.find("figcaption")
            if cap and cap.get_text(strip=True):
                return cap.get_text(" ", strip=True)
        parent = parent.parent
    return ""


def _scrape_html_images(url: str) -> list[Block]:
    try:
        html = _fetch(url, timeout=20)
    except Exception as e:  # noqa: BLE001
        log.debug("image fallback fetch failed for %s: %s", url, e)
        return []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:  # noqa: BLE001
        log.warning("bs4 parse failed: %s", e)
        return []

    seen: set[str] = set()
    out: list[Block] = []
    for img in soup.find_all("img"):
        src = _candidate_src(img)
        if not src:
            continue
        abs_url = urljoin(url, src)
        if abs_url in seen:
            continue
        seen.add(abs_url)
        path_lower = urlparse(abs_url).path.lower()
        ext = path_lower.rsplit(".", 1)[-1] if "." in path_lower else ""
        if ext in _SKIP_EXTS:
            continue
        if any(h in path_lower for h in _SKIP_HINTS):
            continue
        try:
            raw = _fetch(abs_url, timeout=20)
        except Exception as e:  # noqa: BLE001
            log.debug("img fetch failed (%s): %s", abs_url, e)
            continue
        size = len(raw)
        if size < _IMG_MIN_BYTES or size > _IMG_MAX_BYTES:
            continue
        caption = _nearest_caption(img)
        out.append(Block(
            type="image",
            content=caption,
            image_bytes=raw,
            image_ext=ext or "png",
            source_url=url,
            extra={"img_url": abs_url},
        ))
        if len(out) >= _IMG_MAX_PER_PAGE:
            break
    return out
