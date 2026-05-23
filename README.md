# Property-Specific AI Assistant

A chatbot scoped to a single property code (e.g. `115r`) that combines:

- **Structured rent-roll data** (Postgres) — multiple properties × 12 monthly snapshots, unit-level rents, leases, charge-line breakdowns
- **Unstructured marketing content** (Chroma vector store v2) — scraped from property websites + PDFs, extracted with [docling](https://github.com/DS4SD/docling), embedded with the local multimodal **Jina-CLIP-v2** (ONNX) model
- **Runtime LLM switching** across OpenAI, Anthropic, and Google Gemini
- A **LangGraph** agent with 13 bound tools (SQL, RAG, summaries, occupancy, charts, multi-property compare, etc.) and SSE streaming
- A React/Vite frontend that renders Markdown answers plus embedded UI components (KPI cards, tables, line/bar charts via Recharts)

## Architecture (local)

```
[Browser :5173]  React + Vite
        │ /api/* → Vite proxy
        ▼
[FastAPI :8000]  LangGraph agent
        ├── Postgres 16 (docker compose)        — structured rent-roll data
        ├── Chroma      (./chroma_db_v2/)       — vector store, cosine, dim 1024
        ├── doc_store  (./doc_store/)           — extracted images & tables, served at /doc_store/*
        └── LLM APIs   OpenAI / Anthropic / Gemini
```

Embeddings are computed **locally** via ONNX Runtime — no embedding-API costs. Model weights are cached in `HF_HOME`.

## Prerequisites

- Docker Desktop
- Python 3.11+
- Node 18+

## Setup

### 1. Start Postgres

```bash
docker compose up -d
```

Wait ~10s for the healthcheck to pass. On first start, [`backend/db/init_reader.sql`](backend/db/init_reader.sql) provisions the read-only `property_reader` role used by the `execute_scoped_sql` agent tool.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
cp .env.example .env            # fill in LLM API keys
uvicorn app.main:app --reload
```

Then `GET http://localhost:8000/health` should return `{"status":"ok"}`.

### 3. Ingest rent rolls (structured data)

```bash
cd backend
python ingest_all_aker.py
```

This walks `RENT_ROLL_DIR` (set in `.env`), parses the monthly Excel rent rolls, and populates the `properties`, `units`, `leases`, `rent_snapshots`, and `rent_charge_lines` tables.

### 4. Ingest website + PDF content (RAG v2)

The first run downloads the Jina-CLIP-v2 ONNX model (~2 GB) into `HF_HOME`.

```bash
# With the backend running, POST to the admin endpoint:
curl -X POST http://localhost:8000/admin/ingest \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"property_code": "115r", "urls": ["https://example.com/property-page"]}'
```

Pipeline per URL: docling extract → structure-aware chunker → save images/tables to `doc_store/` → embed (text + image) with Jina-CLIP-v2 → upsert into Chroma `property_chunks_v2`.

### 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the FastAPI backend.

## Key env vars

See [`backend/.env.example`](backend/.env.example) for the full list. Highlights:

| Var | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection (docker locally; Supabase pooler on :6543 in prod) |
| `DATABASE_READER_URL` | Read-only `property_reader` role for `execute_scoped_sql` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | LLM providers (set whichever you use) |
| `RAG_VERSION=v2` | Selects the docling + Jina pipeline |
| `CHROMA_DIR_V2`, `COLLECTION_V2` | Local Chroma store location |
| `DOC_STORE_DIR` | Where extracted images/tables are saved |
| `EMBEDDING_MODEL_V2=jinaai/jina-clip-v2`, `EMBEDDING_QUANT=int8\|fp32` | Embedding model & quantization |
| `ADMIN_TOKEN` | Required for `POST /admin/ingest` |
| `RENT_ROLL_DIR` | Path to monthly rent-roll Excel files |

## Layout

```
property-ai-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                 FastAPI app, CORS, /doc_store mount
│   │   ├── config.py               Settings + MODELS registry
│   │   ├── db.py                   SQLAlchemy engine + init_db
│   │   ├── models.py               ORM: properties/units/leases/rent_snapshots/rent_charge_lines (raw_row → JSONB on Postgres)
│   │   ├── schemas.py              Pydantic request/response shapes
│   │   ├── graph/                  LangGraph agent (build.py, nodes.py)
│   │   ├── tools/                  SQL + RAG tools bound to the agent
│   │   ├── guardrails/             Property-scope filter + SQL validator
│   │   └── ingestion/
│   │       ├── rent_roll.py        Excel → Postgres
│   │       └── v2/                 docling pipeline, Jina embedder, Chroma upsert
│   ├── ingest_all_aker.py          One-shot rent-roll loader
│   ├── db/init_reader.sql          Bootstraps read-only role (docker + Supabase)
│   ├── requirements.txt
│   └── .env.example
├── frontend/                       React + Vite UI
├── docker-compose.yml              Postgres 16
└── docs/architecture.md
```

## Deployment

A free-tier cloud deployment plan (Vercel + Supabase + Pinecone + Hugging Face Spaces) is being implemented in subsequent commits.
