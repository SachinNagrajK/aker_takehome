"""Structure-aware chunker.

Rules:
  - Heading blocks update a `section_path` breadcrumb (`H1 > H2 > H3`).
    Text following a heading never crosses back into a prior section.
  - Adjacent text blocks within one section pack up to TARGET_CHARS, with
    OVERLAP_CHARS bleed between successive packed chunks for context.
  - Each table becomes its own chunk (table markdown is the embedded text).
  - Each image becomes its own chunk; the embed text is the caption (or
    the nearest preceding paragraph if no caption).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .extractor import Block


TARGET_CHARS = 800
MIN_CHARS = 80
OVERLAP_CHARS = 120


@dataclass
class Chunk:
    modality: str                       # "text" | "table" | "image"
    embed_input: str                    # text fed to the embedder (markdown for tables, caption for images)
    metadata: dict[str, Any] = field(default_factory=dict)
    image_bytes: bytes | None = None
    image_ext: str | None = None


def _section_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(text for _lvl, text in stack)


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _flush_text(
    out: list[Chunk],
    buf: list[str],
    section_path: str,
    page_title: str,
    source_url: str,
) -> None:
    body = "\n\n".join(b for b in buf if b).strip()
    if len(body) < MIN_CHARS:
        return
    # Slice into chunks of ~TARGET_CHARS with overlap.
    start = 0
    while start < len(body):
        end = min(len(body), start + TARGET_CHARS)
        # Try to end on a paragraph or sentence boundary.
        if end < len(body):
            for sep in ("\n\n", "\n", ". ", " "):
                cut = body.rfind(sep, start + MIN_CHARS, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        piece = body[start:end].strip()
        if len(piece) >= MIN_CHARS:
            out.append(Chunk(
                modality="text",
                embed_input=piece,
                metadata={
                    "section_path": section_path,
                    "page_title": page_title,
                    "url": source_url,
                },
            ))
        if end >= len(body):
            break
        start = max(end - OVERLAP_CHARS, start + 1)


def chunk_blocks(blocks: list[Block]) -> list[Chunk]:
    """Convert ordered Blocks into ordered Chunks ready for embedding."""
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    page_title = ""
    text_buf: list[str] = []
    current_source = ""

    def flush() -> None:
        nonlocal text_buf
        _flush_text(chunks, text_buf, _section_path(heading_stack), page_title, current_source)
        text_buf = []

    last_text_for_caption = ""

    for b in blocks:
        if not current_source:
            current_source = b.source_url

        if b.type == "heading":
            flush()
            if b.extra.get("is_page_title") and not page_title:
                page_title = b.content
                # Page title also seeds the H1 level.
                _push_heading(heading_stack, 1, b.content)
                continue
            _push_heading(heading_stack, b.level or 2, b.content)
            continue

        if b.type == "text":
            text_buf.append(b.content)
            last_text_for_caption = b.content
            continue

        if b.type == "table":
            flush()
            md = (b.content or "").strip()
            if not md and b.html_table:
                md = b.html_table  # fallback
            if not md:
                continue
            chunks.append(Chunk(
                modality="table",
                embed_input=md,
                metadata={
                    "section_path": _section_path(heading_stack),
                    "page_title": page_title,
                    "url": b.source_url,
                    "table_html": b.html_table or "",
                },
            ))
            continue

        if b.type == "image":
            flush()
            caption = (b.content or "").strip() or (last_text_for_caption[:280] if last_text_for_caption else "")
            chunks.append(Chunk(
                modality="image",
                embed_input=caption or page_title or "image",
                metadata={
                    "section_path": _section_path(heading_stack),
                    "page_title": page_title,
                    "url": b.source_url,
                    "caption": caption,
                },
                image_bytes=b.image_bytes,
                image_ext=b.image_ext or "png",
            ))
            continue

    flush()
    # Assign per-document chunk_index for stable ids
    for i, ch in enumerate(chunks):
        ch.metadata["chunk_index"] = i
    return chunks
