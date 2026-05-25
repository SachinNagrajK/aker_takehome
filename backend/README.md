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

## Observability & evaluation

Tracing uses **Phoenix Cloud** (free hosted tier — `https://app.phoenix.arize.com`) via the OpenTelemetry `BatchSpanProcessor`. Spans ship on a background thread, so `/chat` latency is unaffected even when the network or Phoenix is down. Tracing is **opt-in**: set `PHOENIX_ENABLED=true` and `PHOENIX_API_KEY=<key>` in env to turn it on.

Auto-instrumented:
- FastAPI routes
- LangChain / LangGraph nodes, tools, LLM calls (token counts included)
- OpenAI, Anthropic, Google GenAI client SDKs
- Pinecone retrieval (manual span in `tools/rag_tools.py`, OpenInference RETRIEVER kind)

### Evaluation harness

`open_rag_eval` (Vectara, Apache-2.0) scores RAG turns for **groundedness**, **hallucination**, **answer relevance**, and **context relevance**, judged by `gpt-4o-mini` (set via `EVAL_JUDGE_MODEL`). Evals **never** run inline on `/chat` — they are fully out-of-band.

Triggers:
- **Manual** — UI: open the Monitoring tab in the frontend, enter your `ADMIN_TOKEN`, pick cases (or "Run all"), click run. API: `POST /evals/runs` with header `X-Admin-Token`.
- **CLI** — `python -m app.evals.runner [--ids id1,id2]`.
- **Scheduled** — opt-in via `EVAL_SCHEDULE_ENABLED=true` with `EVAL_SCHEDULE_CRON="0 */6 * * *"` (default every 6 h). Runs on an APScheduler `BackgroundScheduler` (single-worker thread pool, coalesce, max_instances=1).

Results land in:
- **Supabase Postgres** — tables `eval_runs` + `eval_cases` (created automatically by `init_db()`). Run history surfaced via the Monitoring UI.
- JSONL snapshots at `backend/evals/results/<timestamp>_<run_id>.jsonl`
- Phoenix Cloud traces under the project `property-ai` in the `aker-ai` space (each case is its own trace; eval scores attached as span attributes)

Extra env:
- `PHOENIX_ENABLED`, `PHOENIX_API_KEY`, `PHOENIX_ENDPOINT`, `PHOENIX_PROJECT_NAME`
- `EVAL_JUDGE_MODEL` (default `gpt-4o-mini`), `EVAL_SCHEDULE_ENABLED`, `EVAL_SCHEDULE_CRON`, `EVAL_MAX_CASES` (default 50)

Edit the golden set at [`app/evals/golden_set.yaml`](app/evals/golden_set.yaml).
