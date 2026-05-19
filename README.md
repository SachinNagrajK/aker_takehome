# Property-Specific AI Assistant

A chatbot scoped to a single property code (e.g. `115r`) that combines:

- **Structured** rent-roll data (MySQL) — 25 properties × 12 monthly snapshots
- **Unstructured** marketing content (Chroma vector store) — scraped from property websites
- Runtime LLM switching across OpenAI, Anthropic, Gemini
- Markdown answers with embedded UI components (KPI cards, tables, charts)

> Status: scaffold + MySQL up. Ingestion, orchestration, UI to follow.

## Prerequisites

- Docker Desktop
- Python 3.11+
- Node 18+

## Setup

### 1. Start MySQL

```bash
docker compose up -d
```

Wait ~10s for the healthcheck to pass.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate         # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
cp .env.example .env           # fill in any LLM API keys you have
uvicorn app.main:app --reload
```

Then `GET http://localhost:8000/health` should return `{"status":"ok"}`.

### 3. (Coming next) Ingest rent rolls + scrape websites + run frontend

Steps wired in upcoming commits.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) and the implementation plan
at the project root.
