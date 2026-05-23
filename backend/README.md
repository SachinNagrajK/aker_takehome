---
title: Aker Property AI
emoji: 🏢
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Aker Property AI — Backend

FastAPI + LangGraph backend for the property-specific AI assistant.

This is the Hugging Face Space deployment of the backend that lives under
[`property-ai-assistant/backend`](https://github.com/SachinNagrajK/aker_takehome/tree/main/property-ai-assistant/backend)
in the source repository. It is the SAME code, just packaged in a Docker
container that HF Spaces runs continuously.

## Wiring

- **Structured data** — Supabase Postgres (Session Pooler)
- **Vector store** — Pinecone serverless (`property-chunks-v2`, namespace per property)
- **Image / table artifacts** — Supabase Storage public bucket (`doc-store`)
- **Embeddings** — Jina-CLIP-v2 ONNX, computed in-process on this container
- **LLMs** — OpenAI / Anthropic / Google Gemini (keys set as Space Secrets)

## Endpoints

- `GET /health` — liveness probe
- `GET /properties` — list of property codes
- `GET /llms` — available LLM providers/models
- `POST /chat` — synchronous chat
- `POST /chat/stream` — Server-Sent Events stream
- `POST /admin/ingest` — re-ingest RAG content (requires `X-Admin-Token`)

## Secrets

All of the following are configured under **Settings → Repository secrets**
on the Space and injected as env vars at runtime:

- `DATABASE_URL`, `DATABASE_READER_URL`
- `PINECONE_API_KEY`, `PINECONE_INDEX`
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- `ADMIN_TOKEN`
- `CORS_ORIGINS` — comma-separated list of allowed Vercel origins
