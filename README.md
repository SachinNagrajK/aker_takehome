# Aker Property AI Assistant

A chatbot scoped to one property at a time (e.g. `134r`). It can:

- Answer questions about the rent roll (occupancy, expiring leases, rent trends, unit comparisons, charge breakdowns) by running SQL against a Postgres database.
- Answer questions about a property's amenities, floor plans, gallery photos, neighborhood, etc. by retrieving from a Pinecone vector store. The store was populated from each property's marketing website using docling (text + table + image extraction) and Jina-CLIP-v2 embeddings (multimodal, hosted on Hugging Face).
- Switch LLM provider at runtime: OpenAI, Anthropic, or Google Gemini.
- Render KPI cards, tables, and charts inline in the chat (line, bar, pie, comparison).
- Stream the agent's reasoning and tool calls back to the UI as Server-Sent Events.
- Run an automated eval harness against a hand-curated golden set, score with an LLM judge on four metrics, and surface everything in a Monitoring tab.

The agent is built on LangGraph with 13 bound tools. It pauses (via `interrupt()`) when scope is ambiguous and re-routes through a clarify node before answering.

---

## Architecture

```
Browser (React + Vite + Recharts)
    │  HTTP and SSE  (Vite proxies /api/* to FastAPI)
    ▼
FastAPI  (Hugging Face Spaces in prod, local uvicorn in dev)
    │
    ▼
LangGraph state machine
  extract_scope -> clarify or clarify_time (interrupt) -> enter_turn
                -> agent <-> tools -> compose
    │
    ├─> Tool layer (13 tools)
    │     SQL tools  -> Supabase Postgres (read-only role)
    │     RAG tools  -> Pinecone (cosine, 1024-d)
    │     render_chart -> structured components[] back to the UI
    │
    ├─> Guardrails
    │     scope.py            rejects any tool call missing property_code
    │     sql_validator.py    allowlist + scope check before SQL runs
    │
    ├─> Embeddings   Jina-CLIP-v2 hosted on Hugging Face
    ├─> LLMs         OpenAI, Anthropic, Gemini
    └─> Phoenix Cloud (OTLP, batched, async, fail-open)
```

### Components

- **Frontend** — React with Vite and Recharts. The chat renders streamed Markdown plus a structured `components[]` array (KPI card, line / bar / pie / comparison charts, data table, image gallery, lightbox, tool trace, clarification card, monitoring panel).
- **API** — FastAPI. Routes: `/chat` (SSE), `/admin/ingest`, `/evals/*`, `/properties`, `/llms`, `/health`.
- **Agent** — LangGraph state machine over `ChatState`. Uses `InMemorySaver` keyed by `conversation_id`, so `interrupt()` for clarifications can resume cleanly when the user replies.
- **Tools (13)** — `get_property_summary`, `get_unit_mix`, `get_occupancy`, `get_rent_trend`, `get_expiring_leases`, `get_top_balances`, `get_lease_deposits`, `get_move_outs`, `get_unit_charges`, `compare_units`, `list_units`, `execute_scoped_sql`, `render_chart`, plus `search_property_pages` for RAG.
- **Guardrails** — every tool call needs a resolved `property_code`. The custom-SQL backstop runs through an allowlist and a scope-predicate check before it ever hits the database, and the database connection itself is a SELECT-only role.
- **Stores** — Postgres for structured data (`properties -> units -> leases -> rent_snapshots -> rent_charge_lines`) and Pinecone for vectors.
- **Observability** — Phoenix Cloud via OpenInference. LangChain, the LLM SDKs, and FastAPI are auto-instrumented. Spans are exported with `BatchSpanProcessor` so the chat path never blocks on telemetry. If `PHOENIX_API_KEY` isn't set, tracing is a no-op.
- **Evaluation** — `golden_set.yaml` drives the live agent end-to-end through `runner.py`. An LLM judge in `scorer.py` returns four scores on a 0.25-step scale. Results land in SQLite and a JSONL log. APScheduler runs the harness on a cron, and the Monitoring tab in the UI surfaces all of it.

### Request lifecycle

1. UI posts `{property_code, message, llm_provider, model, conversation_id}` to `/chat`.
2. `extract_scope` reconciles the property dropdown with anything named in free text. If scope is missing or conflicts, the graph pauses at `clarify` and sends an SSE `clarification` event.
3. `enter_turn` seeds the system prompt with the resolved scope.
4. `agent` loops with the chosen LLM. Each tool call emits a `tool` event with a human-readable reasoning line ("Loading rent trend · 115r").
5. The `tools` node runs the call. The result goes into `tool_history` and a `tool_end` event reports `ok` and `duration_ms`.
6. `compose` flushes the final Markdown and `components[]` and the stream ends with `done`.

---

## Setup

### Prerequisites

- Docker Desktop (running)
- Python 3.11+
- Node 18+
- An OpenAI API key. Anthropic and Gemini are optional.

### 1. Start Postgres

```bash
docker compose up -d
docker compose ps     # confirm property_ai_postgres is "healthy"
```

On first start, `backend/db/init_reader.sql` creates the read-only `property_reader` role used by the SQL executor.

You can skip Docker and point `DATABASE_URL` and `DATABASE_READER_URL` at a Supabase pooler (port `6543`) instead. Same code path.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
cp .env.example .env             # fill in keys (table below)
uvicorn app.main:app --reload
```

Then `GET http://localhost:8000/health` should return `{"status":"ok"}`.

### 3. Ingest rent rolls (structured data)

```bash
python -m app.ingestion.rent_roll
```

This walks the directory in `RENT_ROLL_DIR`, parses every monthly Excel rent roll, and populates `properties`, `units`, `leases`, `rent_snapshots`, and `rent_charge_lines`.

### 4. Ingest website and PDF content (RAG)

Embeddings are computed by calling the Jina-CLIP-v2 model hosted on Hugging Face. Nothing is downloaded locally. Vectors land in Pinecone. Images and table renderings land in Supabase Storage.

Seed all Aker portfolio properties in one shot:

```bash
python ingest_all_aker.py
```

Or ingest one property on demand:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "X-Admin-Token: $ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"property_code":"134r","urls":["https://example.com/property-page"]}'
```

Per URL: docling extracts text, tables, and images. A structure-aware chunker turns them into chunks. Images and tables are saved to Supabase Storage. Text and image chunks are embedded via Jina-CLIP-v2 on Hugging Face. Everything is upserted into Pinecone (`property-chunks-v2`).

> **RAG coverage note.** I scraped marketing-site content for 10 of the 22 properties. Pick one of these in the dropdown if you want to exercise the RAG path (amenities, photos, neighborhood, floor plans):
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
> Other properties still answer structured questions normally. They just say "no marketing content ingested for this property" and fall back to SQL for amenity-style questions.

### 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api/*` to the FastAPI backend.

### 6. Optional: enable monitoring and evals

- **Tracing**: set `PHOENIX_ENABLED=true` and `PHOENIX_API_KEY=...` and restart.
- **Scheduled evals**: set `EVAL_SCHEDULE_ENABLED=true`, `EVAL_SCHEDULE_CRON="0 */6 * * *"`, `EVAL_JUDGE_MODEL=gpt-4o-mini`.
- **Manual run from the UI**: open the Monitoring tab, paste the admin token, pick cases, click Run.
- **From the command line**:
  ```bash
  python -m app.evals.runner --provider openai
  ```

### Smoke-test queries

The app loads with `134r` (Fifty-Five Riverwalk Place) preselected. It has 99% per-unit rent coverage in source and the largest RAG corpus (292 chunks), so every tool path is exercised on first load. Try:

- "Give me a summary of this property." — exercises `get_property_summary`.
- "Show the rent trend over the last 12 months." — exercises `get_rent_trend` and `render_chart`.
- "List leases expiring in the next 90 days." — exercises `get_expiring_leases`.
- "Compare any two units." — exercises `list_units`, then `compare_units`, then `render_chart`.
- "What amenities does this property offer? Show a few photos." — exercises the RAG path with multimodal image retrieval.
- "Who is moving out soon?" — exercises `get_move_outs`.
- "What's the occupancy?" with no property selected — triggers a property-scope clarification.
- "Which units have the highest balance?" — triggers a time-scope clarification (latest vs specific month).

### Key env vars

The full list is in [`backend/.env.example`](backend/.env.example).

| Var | Purpose |
|---|---|
| `DATABASE_URL`, `DATABASE_READER_URL` | Postgres connection strings. The reader URL points to a SELECT-only role used by the LLM SQL executor. |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | LLM providers. Only the ones you supply show up as available in the model dropdown. |
| `PINECONE_API_KEY`, `PINECONE_INDEX` | Vector store. |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET` | Where extracted images and tables live. |
| `EMBEDDING_MODEL_V2=jinaai/jina-clip-v2` | Hugging Face model id for embeddings. |
| `ADMIN_TOKEN` | Required for `/admin/*` and `/evals/*`. |
| `PHOENIX_ENABLED`, `PHOENIX_API_KEY` | Phoenix Cloud tracing. Fail-open if unset. |
| `EVAL_SCHEDULE_ENABLED`, `EVAL_SCHEDULE_CRON`, `EVAL_JUDGE_MODEL` | APScheduler cron evals. |
| `RENT_ROLL_DIR` | Path to the monthly rent-roll Excel files. |

---

## Design decisions

A few choices worth flagging:

- **LangGraph instead of a plain ReAct loop.** I wanted `interrupt()` for clarifications and a checkpointer for resumable conversations. LangGraph gives both.
- **Scope is a graph concept, not a prompt hint.** Tool wrappers reject calls without a resolved `property_code`. The agent cannot accidentally answer about the wrong property.
- **Read-only Postgres role plus SQL validator.** Even if the LLM emits something nasty through `execute_scoped_sql`, the role lacks `INSERT/UPDATE/DELETE/DDL`, and the validator rejects multi-statement queries and anything missing a `property_code` predicate.
- **Structured components instead of free-form chart JSON.** The LLM declares *what* to render via a small `components[]` schema. The backend supplies the data from SQL tools. The frontend renders fixed React components. The user never sees a chart the LLM made up.
- **Jina-CLIP-v2 on Hugging Face.** Multimodal (text and image in the same 1024-d space) and hosted, so the backend Docker image stays small. One network hop per embedding is the tradeoff.
- **Pinecone instead of local Chroma.** Chroma was perfect during local iteration. Pinecone is what survives on a free-tier Space without me babysitting a sqlite file on disk.
- **Per-request LLM switching.** `llm_registry.py` treats provider and model as request parameters. Tool surface is identical across OpenAI, Anthropic, and Gemini.
- **Typed SSE events instead of a single spinner.** Events: `step`, `tool`, `tool_end`, `delta`, `clarification`, `done`, `error`. The user sees the agent's tool calls live, which is much better than staring at a loader for 30 seconds.
- **Phoenix with OpenInference.** Vendor-neutral OTel. Auto-instruments LangChain. Batched export. If the key is missing, tracing turns into a no-op rather than throwing.
- **In-process APScheduler for evals.** One thread pool, `coalesce=True`, `max_instances=1`. No separate worker to operate. Eval runs never touch the request thread pool.

---

## Tradeoffs and limitations

**Tradeoffs.**

- `InMemorySaver` for checkpointing. Conversations evaporate on backend restart. A Postgres saver is a one-line swap when it matters.
- Hugging Face hosted embeddings. Backend image stays slim and I don't ship ~2 GB of model weights. The cost is one network round-trip per embedding and a hard HF dependency.
- Chroma to Pinecone. Extra vendor, but I don't have to babysit a file-backed store on a free-tier Space.
- The SQL validator's table allowlist will sometimes false-reject legitimate LLM SQL (e.g. a clever CTE). I accept that to keep the blast radius small.
- SSE instead of WebSockets. One-way is enough for chat and it survives every reverse proxy I tested.
- LLM-as-judge for evals. Cheap and consistent, but a same-family judge can self-collude. The 0.25-step rubric, the requirement to list concrete issues, and the "default to 0.75 if uncertain" line in the prompt all try to keep the judge honest.
- APScheduler in-process. Simple. Eventually a dedicated worker would be the right call if eval runs ever compete with chat traffic.
- I kept the 8 properties that have no per-unit rent data in the source files (`153a`, `153r`, `175r`, `176r`, `183a`, `183r`, `184r`, `185r`). I considered hiding them from the dropdown, but the task said to use the dataset you provided, so I kept the demo honest. The default property is now `134r` (full data) so the first-load experience isn't degraded. If you pick one of the 8, you'll see market rent only, and the agent says so.

**Limitations.**

- *Source data gap on 8 properties.* These codes have full rows in `properties`, `units`, `leases`, and `rent_snapshots`, but every lease has `monthly_rent = NULL` and there are zero `rent_charge_lines`. That's because the Aker-side rent-roll export for those workbooks was generated without the per-unit charge subdetail. Only "Market Rent" totals are in the file. The bottom-of-workbook aggregate line literally reads `Lease Charges: 0.00`.

  | Code | Property                       | Leases | RAG chunks | Per-unit rent in source? |
  |------|--------------------------------|-------:|-----------:|:------------------------:|
  | 153a | Abbot Mill (affordable)        |     18 |      —     | ❌                       |
  | 153r | Abbot Mill                     |    192 |     177    | ❌                       |
  | 175r | Kinwood Apartments             |    358 |     228    | ❌                       |
  | 176r | The Alexander                  |    262 |     267    | ❌                       |
  | 183a | Luckey Platt (affordable)      |     24 |      —     | ❌                       |
  | 183r | Luckey Platt                   |    105 |     253    | ❌                       |
  | 184r | Lakeshore Preserve             |    134 |     219    | ❌                       |
  | 185r | Waterfront at the Strand       |     58 |     282    | ❌                       |

  Verified by reading the raw `.xls` files. `Dec_RENT_ROLL_WITH_LEASE_CHARGES_175r.xls` line 1091 shows `Totals: Market Rent 771,397.00, Lease Charges 0.00`. `validate_ingestion.py` confirms 300/300 workbooks ingested cleanly across 94,584 charge-line rows. There was simply nothing to extract for these 8 properties. For comparison, `115r`, `134r`, `462a` and other "full-data" properties have per-unit RENT, PETFEEM, AMENITY, PARKING, TRASH charge codes.
- No auth on `/chat`. The deployed URL is shared by obscurity. `ADMIN_TOKEN` only guards `/admin/*` and `/evals/*`.
- The golden eval set is hand-curated and small (about 10 cases). Broader coverage and adversarial cases are the next step.
- RAG quality depends entirely on what was ingested per property. When a page wasn't ingested, the agent falls back to SQL and says so.
- No per-request LLM-cost cap. Budgets are enforced upstream by the provider.
- Hugging Face Spaces has a cold-start hit on the first request after idle.
- `compare_properties` is currently disabled. I didn't have a good UX for comparing more than two properties at once.

---

## Deployment

The app runs across free tiers stitched together:

- **Frontend** — Vercel (Vite build, edge-served).
- **Backend** — Hugging Face Spaces (Docker SDK Space running FastAPI). See [`backend/README.md`](backend/README.md) for the Space card and required secrets.
- **Postgres** — Supabase. Session Pooler on port `6543`. `init_reader.sql` provisions the same read-only role used locally.
- **Vector store** — Pinecone serverless. Index `property-chunks-v2`, cosine, 1024-d.
- **Object store** — Supabase Storage public bucket `doc-store` for extracted images and tables.
- **Telemetry** — Phoenix Cloud (Arize) over OTLP, batched, fail-open.

---

## Layout

```
property-ai-assistant/
├── README.md                       (this file)
├── docker-compose.yml              Postgres 16 for local dev
├── backend/
│   ├── README.md                   HF Space card (required by Hugging Face)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── main.py                 FastAPI app, CORS, observability bootstrap
│   │   ├── config.py               Settings, MODELS registry
│   │   ├── db.py                   SQLAlchemy engine, init_db
│   │   ├── models.py               ORM (properties, units, leases, rent_snapshots, rent_charge_lines)
│   │   ├── schemas.py              Pydantic request/response shapes
│   │   ├── llm_registry.py         OpenAI / Anthropic / Gemini switching per request
│   │   ├── observability.py        Phoenix Cloud, OpenInference (fail-open)
│   │   ├── graph/                  LangGraph agent (build.py, nodes.py)
│   │   ├── tools/                  SQL and RAG tools bound to the agent
│   │   ├── guardrails/             Property-scope filter and SQL validator
│   │   ├── evals/                  Golden set, runner, scorer, scheduler, admin API
│   │   └── ingestion/
│   │       ├── rent_roll.py        Excel to Postgres
│   │       └── v2/                 docling pipeline, HF embedder, Pinecone upsert
│   ├── ingest_all_aker.py          One-shot RAG seed for the Aker portfolio
│   ├── db/init_reader.sql          Bootstraps the read-only role (docker + Supabase)
│   ├── wipe_and_reingest.py        Utility
│   ├── validate_ingestion.py       Utility
│   └── upload_doc_store_to_supabase.py   One-shot migration utility
└── frontend/                       React + Vite UI (chat plus Monitoring tab)
```

---

## Reviewer pointers

If you want to skim the code, the interesting bits are:

- `backend/app/graph/build.py`, `backend/app/graph/nodes.py` — the agent topology.
- `backend/app/tools/sql_tools.py`, `backend/app/tools/rag_tools.py` — the 13 tools.
- `backend/app/guardrails/` — scope filter and SQL validator.
- `backend/app/ingestion/v2/` — docling extractor, chunker, embedder, Pinecone upsert.
- `backend/app/observability.py` — Phoenix wiring.
- `backend/app/evals/` — runner, scorer, scheduler, admin API, golden set.
- `frontend/src/components/Monitoring.jsx` — the eval admin UI.
- `frontend/src/components/ComponentRenderer.jsx` — the `components[]` dispatcher.
