"""RAG v2 — docling-based multimodal ingestion + Jina v4 embeddings.

Public entry point: `pipeline.ingest_urls(property_code, urls, replace=False)`.

The v1 pipeline (`app.ingestion.scrape_and_index`) is left untouched so it
remains available as a rollback path via `RAG_VERSION=v1`.
"""
