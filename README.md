# Property-Specific AI Assistant

A chatbot scoped to a single property code (e.g. `115r`) that fuses:

- **Structured rent-roll data** (Postgres) — multiple properties × 12 monthly snapshots, unit-level rents, leases, charge-line breakdowns
- **Unstructured marketing content** (Pinecone, 1024-d cosine) — scraped from property websites + PDFs, extracted with [docling](https://github.com/DS4SD/docling), embedded with the multimodal **Jina-CLIP-v2** model hosted on Hugging Face *(v1 used a local Chroma store — kept in the repo for offline iteration only)*
- **Runtime LLM switching** across OpenAI, Anthropic, and Google Gemini
- A **LangGraph** agent with 13 bound tools (SQL, RAG, summaries, occupancy, charts, multi-property compare, …) and SSE streaming
- A React/Vite frontend that renders Markdown answers plus embedded UI components (KPI cards, tables, line/bar/pie charts via Recharts)
- **Observability** via Phoenix Cloud + an **automated eval harness** with golden-set regression scoring (groundedness / hallucination / answer-relevance / context-relevance)

---

## Architecture

```
Browser (React + Vite + Recharts)
    │  HTTP + SSE  (/api → Vite proxy → FastAPI)
    ▼
FastAPI  (Hugging Face Space in prod, local uvicorn in dev)
    │
    ▼
LangGraph state machine
  extract_scope → clarify / clarify_time (interrupt)
                → enter_turn → agent ⇄ tools → compose
    │
    ├──► Tool layer (13 tools)
    │      • SQL tools          → Supabase Postgres (read-only role)
    │      • RAG tools          → Pinecone (1024-d cosine)
    │      • render_chart       → structured components[]
    │
    ├──► Guardrails
    │      • scope.py           rejects calls missing property_code
    │      • sql_validator.py   allowlist + scope predicate before SQL runs
    │
    ├──► Embeddings (Jina-CLIP-v2, hosted on Hugging Face)
    ├──► LLM providers (OpenAI / Anthropic / Gemini)
    └──► Phoenix Cloud  (OTLP, batched, async)
```

### Components

- **Frontend** — React + Vite + Recharts. Renders streamed Markdown plus a structured `components[]` array (`KPICard`, `LineChartComp`, `BarChartComp`, `PieChartComp`, `ComparisonChart`, `DataTable`, `ImageGallery`, `Lightbox`, `ToolTrace`, `ClarificationCard`, `Monitoring`).
- **API** — FastAPI: `/chat` (SSE), `/admin/ingest`, `/evals/*`, `/properties`, `/llms`, `/health`.
- **Agent** — LangGraph state machine over `ChatState`; `InMemorySaver` keyed by `conversation_id` so `interrupt()` for clarifications resumes cleanly.
- **Tools (13)** — `get_property_summary`, `get_unit_mix`, `get_occupancy`, `get_rent_trend`, `get_expiring_leases`, `get_top_balances`, `get_lease_deposits`, `get_move_outs`, `get_unit_charges`, `compare_units`, `list_units`, `execute_scoped_sql`, `render_chart`, plus `search_property_pages` / `search_property_active` for RAG.
- **Guardrails** — every tool call carries a resolved `property_code`; custom SQL goes through an allowlist + scope-predicate check before the read-only role executes it.
- **Stores** — Postgres 16 for structured data (`properties → units → leases → rent_snapshots → rent_charge_lines`); **Pinecone** for vectors.
- **Observability** — Phoenix Cloud via OpenInference; LangChain / LLM SDK / FastAPI auto-instrumented; `BatchSpanProcessor` so export never blocks the request hot path; fail-open if no key.
- **Evaluation** — curated `golden_set.yaml` → `runner.py` drives the live graph end-to-end → `scorer.py` LLM-as-judge returns four metrics on a 0.25-step scale → JSONL + SQLite persistence → APScheduler cron + admin API + **Monitoring** tab in the UI.

### Request lifecycle

1. UI posts `{property_code, message, llm_provider, model, conversation_id}` to `/chat`.
2. `extract_scope` reconciles the dropdown with anything named in free text; missing/conflicting scope → `clarify` interrupt → SSE `clarification` event.
3. `enter_turn` seeds the system prompt with the resolved scope.
4. `agent` loops with the chosen LLM; each tool call streams a `tool` event with a human-readable reasoning line (*"Loading rent trend · 115r"*).
5. `tools` node executes; results land in `tool_history`, streamed as `tool_end` with `ok` / `duration_ms`.
6. `compose` flushes the final Markdown + `components[]` and ends with `done`.

---

## Setup

### Prerequisites

- Docker Desktop (running)
- Python 3.11+
- Node 18+
- An OpenAI API key (Anthropic / Gemini optional)

### 1. Start Postgres

```bash
docker compose up -d
docker compose ps     # confirm property_ai_postgres is "healthy"
```

On first start, `backend/db/init_reader.sql` provisions the read-only `property_reader` role used by the SQL executor.

*Alternative:* skip Docker and point `DATABASE_URL` / `DATABASE_READER_URL` at a Supabase pooler (port `6543`) instead. The same code path runs either way.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
cp .env.example .env             # fill in keys (see table below)
uvicorn app.main:app --reload
```

Verify: `GET http://localhost:8000/health` → `{"status":"ok"}`.

### 3. Ingest rent rolls (structured data)

```bash
python -m app.ingestion.rent_roll
```

Walks `RENT_ROLL_DIR` (set in `.env`), parses the monthly Excel rent rolls, and populates `properties / units / leases / rent_snapshots / rent_charge_lines`.

### 4. Ingest website + PDF content (RAG)

Embeddings are computed by calling the **Jina-CLIP-v2** model hosted on Hugging Face (no local model download). Vectors land in **Pinecone**; extracted images/tables land in **Supabase Storage**.

Seed all Aker portfolio property pages in one shot:

```bash
python ingest_all_aker.py
```

Or ingest a single property on demand:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "X-Admin-Token: $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"property_code":"115r","urls":["https://example.com/property-page"]}'
```

Pipeline per URL: docling extract → structure-aware chunker → save images/tables to Supabase Storage → embed (text + image) via HF-hosted Jina-CLIP-v2 → upsert into Pinecone (`property-chunks-v2`).

> **RAG coverage note.** Only **10 properties** have been ingested into the vector store so far. Pick one of these in the property dropdown if you want to exercise the RAG path (questions about amenities, photos, neighborhood, floor plans, etc.):
>
> | Code | Property |
> |------|----------|
> | `134r` | Fifty-Five Riverwalk Place |
> | `138r` | Everbend Tarrytown |
> | `139r` | The Mill Greenwich |
> | `153r` | Abbot Mill |
> | `175r` | Kinwood Apartments |
> | `176r` | The Alexander |
> | `183r` | Luckey Platt |
> | `184r` | Lakeshore Preserve |
> | `185r` | Waterfront at the Strand |
> | `462a` | Stony Run |
>
> Any other property still answers structured (SQL) questions normally — only RAG / image lookups will say "no marketing content ingested for this property" and fall back to SQL.

### 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the FastAPI backend.

### 6. (Optional) Enable monitoring & evals

- **Tracing**: set `PHOENIX_ENABLED=true` and `PHOENIX_API_KEY=…`, restart.
- **Scheduled evals**: set `EVAL_SCHEDULE_ENABLED=true`, `EVAL_SCHEDULE_CRON="0 */6 * * *"`, `EVAL_JUDGE_MODEL=gpt-4o-mini`.
- **Manual run**: open the **Monitoring** tab, paste the admin token, pick cases, click Run. Or:
  ```bash
  python -m app.evals.runner --provider openai
  ```

### Smoke-test queries

The app loads with `175r` (Kinwood Apartments) preselected — it has both structured rent-roll data (358 units, the largest in the portfolio) and RAG-ingested marketing content, so every tool path is exercised by default. Try:

- *"Give me a summary of this property."* — exercises `get_property_summary`.
- *"Show the rent trend over the last 12 months."* — exercises `get_rent_trend` + `render_chart`.
- *"List leases expiring in the next 90 days."* — exercises `get_expiring_leases`.
- *"What amenities does this property offer?"* — exercises `search_property_pages` (RAG).
- *"Show me photos of the pool."* — exercises the multimodal image retrieval.
- *"What's the occupancy?"* after clearing the property selection — triggers a clarification.

### Key env vars

See [`backend/.env.example`](backend/.env.example) for the full list.

| Var | Purpose |
|---|---|
| `DATABASE_URL`, `DATABASE_READER_URL` | Postgres (app role + read-only role for LLM-written SQL) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | LLM providers |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | Vector store (production) |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET` | Image / table artifact storage |
| `EMBEDDING_MODEL_V2=jinaai/jina-clip-v2` | Hugging Face model id for the embedder |
| `ADMIN_TOKEN` | Required for `/admin/*` and `/evals/*` |
| `PHOENIX_ENABLED`, `PHOENIX_API_KEY` | OpenTelemetry export to Phoenix Cloud (fail-open) |
| `EVAL_SCHEDULE_ENABLED`, `EVAL_SCHEDULE_CRON`, `EVAL_JUDGE_MODEL` | APScheduler cron evals |
| `RENT_ROLL_DIR` | Path to monthly rent-roll Excel files |

---

## Design decisions

- **LangGraph over plain ReAct** — first-class `interrupt()` for clarifications and a checkpointer for resumable conversations, both for free.
- **Scope as a first-class graph concept** — tool wrappers reject calls without a resolved `property_code`. The agent literally cannot answer about the wrong property by accident.
- **Read-only Postgres role + allowlist SQL validator** — even if the LLM emits malicious SQL through `execute_scoped_sql`, the role lacks `INSERT/UPDATE/DELETE/DDL` and the validator rejects multi-statement queries and any statement missing a `property_code` predicate.
- **Trusted UI from structured intent** — the LLM emits `components[]` declaring *what* to show; the backend supplies the actual data from SQL tools; the frontend renders fixed React components. The user never sees raw chart JSON the LLM made up.
- **HF-hosted Jina-CLIP-v2 embeddings** — multimodal (text + image in one 1024-d space). Hosting on Hugging Face keeps the backend image slim, at the cost of one network hop per embedding.
- **Chroma → Pinecone migration** — v1 used local Chroma (zero config, perfect for offline iteration); v2 moved to Pinecone for the deployed app (managed, no disk to babysit on HF Spaces, free tier sufficient).
- **Runtime LLM switching** — `llm_registry.py` treats provider × model as a per-request choice; same tool surface across OpenAI, Anthropic, Gemini.
- **Typed SSE event protocol** — `step / tool / tool_end / delta / clarification / done / error`. Materially improves perceived latency on multi-tool turns vs. a single spinner.
- **Phoenix + OpenInference** — vendor-agnostic OTel, instrument-the-library, fail-open so a missing key never blocks `/chat`.
- **APScheduler in-process for evals** — single thread pool, `coalesce=True`, `max_instances=1`; no separate worker to operate; never touches the request thread pool.

---

## Tradeoffs & limitations

**Tradeoffs**
- *`InMemorySaver` checkpointing* — conversations evaporate on restart; a Postgres saver is a one-line swap.
- *HF-hosted embeddings* — keeps the backend image slim and avoids bundling ~2 GB of weights, at the cost of one network round-trip per embedding and a hard HF dependency.
- *Chroma → Pinecone* — extra vendor in exchange for not babysitting a file-backed store on a free-tier Space.
- *Allowlist SQL validator* — false-rejects some legitimate LLM SQL; we accept that to bound blast radius.
- *SSE over WebSockets* — one-way is enough for chat and survives every reverse proxy we tested.
- *LLM-as-judge for evals* — cheap and comparable, but can self-collude with same-family models; the 0.25-step rubric + concrete-issue requirement + default-to-0.75 mitigate.
- *APScheduler in-process* — simple, but shares the FastAPI VM; a dedicated worker is the next step if eval runs ever compete with request traffic.

**Limitations / known gaps**
- No auth on `/chat`; the deployed URL is shared by obscurity. `ADMIN_TOKEN` guards `/admin/*` and `/evals/*` only.
- Golden set is hand-curated (~10 cases) — broader coverage and adversarial cases are the obvious next step.
- RAG quality depends on what was ingested per property; the agent falls back to SQL and says so when a page wasn't ingested.
- No per-request LLM-cost cap; budgets are enforced upstream by the provider.
- HF Spaces cold-start latency on the first request after idle.
- `compare_properties` is currently disabled pending UX for >2 properties.

---

## Deployment

The app runs across free tiers stitched together:

- **Frontend** — Vercel (React/Vite build, edge-served)
- **Backend** — Hugging Face Spaces (FastAPI + LangGraph in a Docker SDK Space; see [`backend/README.md`](backend/README.md) for the Space card / required secrets)
- **Postgres** — Supabase (Session Pooler on port `6543`; `init_reader.sql` provisions the same read-only role used locally)
- **Vector store** — Pinecone serverless (index `property-chunks-v2`, cosine, 1024-d)
- **Object store** — Supabase Storage public bucket `doc-store` for extracted images/tables
- **Telemetry** — Phoenix Cloud (Arize) over OTLP, batched + fail-open

---

## Layout

```
property-ai-assistant/
├── README.md                       ← you are here (setup + architecture + design + tradeoffs)
├── docker-compose.yml              Postgres 16
├── backend/
│   ├── README.md                   HF Space card (required by Hugging Face)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py                 FastAPI app, CORS, observability bootstrap
│   │   ├── config.py               Settings + MODELS registry
│   │   ├── db.py                   SQLAlchemy engine + init_db
│   │   ├── models.py               ORM (properties/units/leases/rent_snapshots/rent_charge_lines)
│   │   ├── schemas.py              Pydantic request/response shapes
│   │   ├── llm_registry.py         OpenAI / Anthropic / Gemini per-request switching
│   │   ├── observability.py        Phoenix Cloud + OpenInference (fail-open)
│   │   ├── graph/                  LangGraph agent (build.py, nodes.py)
│   │   ├── tools/                  SQL + RAG tools bound to the agent
│   │   ├── guardrails/             Property-scope filter + SQL validator
│   │   ├── evals/                  Golden set, runner, scorer, scheduler, admin API
│   │   └── ingestion/
│   │       ├── rent_roll.py        Excel → Postgres
│   │       └── v2/                 docling pipeline, HF embedder, Pinecone upsert
│   ├── ingest_all_aker.py          One-shot RAG seed for the Aker portfolio
│   ├── db/init_reader.sql          Bootstraps read-only role (docker + Supabase)
│   ├── wipe_and_reingest.py        Utility
│   ├── validate_ingestion.py       Utility
│   └── upload_doc_store_to_supabase.py   One-shot migration utility
├── frontend/                       React + Vite UI (incl. Monitoring tab)
└── report/                         LaTeX report
    ├── REPORT.tex                  Single-file source, Overleaf-ready
    ├── REPORT.pdf                  Built output
    └── screenshots/                Drop PNGs here (see screenshots/README.md)
```

---

## Reviewer pointers

- `backend/app/graph/build.py`, `graph/nodes.py` — agent topology.
- `backend/app/tools/sql_tools.py`, `rag_tools.py` — 13 tools.
- `backend/app/guardrails/` — scope filter + SQL validator.
- `backend/app/ingestion/v2/` — docling extractor, chunker, embedder, Pinecone upsert.
- `backend/app/observability.py` — Phoenix wiring.
- `backend/app/evals/` — runner, scorer, scheduler, API, golden set.
- `frontend/src/components/Monitoring.jsx` — eval admin UI.
- `frontend/src/components/ComponentRenderer.jsx` — `components[]` dispatch.

---

