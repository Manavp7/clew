# clew — Canonical Intelligence Engine

A domain-agnostic intelligence engine that turns unstructured documents into a
**trustworthy, time-aware, evidence-backed knowledge graph**.

> **Core thesis:** store *evidence-backed claims over time* — not facts —
> resolved to *canonical entities*, and measure everything.

## Architecture (one decision everything hangs on)

The single source of truth is an **append-only, bitemporal Claim Ledger** in
Postgres. The graph and vector index are **rebuildable projections** of it.

```
 Documents → Ingest → Parse → Extract(+offsets) → Entity Resolution
   → CLAIM LEDGER (Postgres, source of truth)
   → project → Graph (NetworkX) + Vectors (pgvector) → Reasoning → Cited output
```

Binding constraints:

1. The Claim Ledger is the **only** source of truth.
2. Graph / vector / GraphRAG indexes are **projections**, rebuildable anytime.
3. The reasoning layer is **read-only** — it cannot write claims.
4. **Only extraction pipelines** create or update claims.
5. Every claim carries **≥1 evidence span + extractor version**.
6. The **evaluation harness is first-class**, built alongside extraction & ER.

## Launch wedge

**Pack A — Investigative/Financial**, starting with **SEC EDGAR 13D/13G
beneficial-ownership filings**: ER, temporal claims, supersession, ownership
changes, contradiction detection, and stake-change timelines.

## Quickstart

```bash
# 1. Install deps (uv)
uv sync --extra dev

# 2. Start Postgres + apply the ledger schema
make db-up && make migrate        # native Postgres 16 (apt)
# or, via Docker:  make db-up-docker && make migrate

# 3. Run tests
make test
```

Copy `.env.example` to `.env` and set `CLEW_LLM_API_KEY` (Vercel AI Gateway) for
live claim extraction.

## Tech

Postgres 16 + pgvector · SQLAlchemy/Alembic · `uv` · GLiNER · Splink · NetworkX ·
FastAPI · Vercel AI Gateway (provider-agnostic LLM access).

### Pluggable backends & deferred infra

- **Vector store** (`CLEW_VECTOR_BACKEND`): `pgvector` (default, in Postgres) or
  `qdrant` (embedded, on-disk via `qdrant-client` — no server).
- **Graph** is a NetworkX projection (Neo4j intentionally not used yet; the ledger
  is the source of truth and the graph is rebuildable).
- **Orchestration**: an idempotent `clew` CLI pipeline. **Temporal is deferred** —
  it needs a server cluster and is unnecessary at single-machine scale; it's the
  future durable-workflow upgrade when scale/throughput demands it.
