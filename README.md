# AI IT Incident Resolution Agent

An observable, controlled incident workflow built with Python, LangGraph, FastAPI, and uv. It uses simulated enterprise tools only; no production systems or credentials are required.

## Workflow

`incident -> triage -> parallel-style tool investigation -> diagnosis -> risk routing -> approval (high risk) -> simulated remediation -> verification -> replan/retry -> structured report`

The compiled LangGraph state machine demonstrates sequential nodes, conditional routing, tool calling, human-in-the-loop approval, and retry/replanning. The payment scenario intentionally fails its first health check to exercise the recovery path.

## Run locally

```bash
uv sync
uv run uvicorn it_agents.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

### API examples

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/incidents \
  -H 'Content-Type: application/json' \
  -d '{"incident_id":"INC-1042","service":"payment-service","description":"Customers report payment failures","error":"Database connection timeout"}'
curl -X POST http://127.0.0.1:8000/incidents/INC-1042/approval \
  -H 'Content-Type: application/json' -d '{"approved":true}'
curl http://127.0.0.1:8000/incidents/INC-1042
```

High-risk incidents return `awaiting_approval`; call the approval endpoint before execution. Low-risk incidents complete automatically.

## Tests and container

```bash
uv run pytest -q
docker compose up --build
```

All remediation tools are deterministic simulations (`search_logs`, `get_service_metrics`, `search_knowledge_base`, `get_incident_history`, `restart_service`, and `check_service_health`). When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in a local `.env`, each resolution is emitted as a Langfuse chain observation; without credentials the wrapper is a safe no-op. `.env.example` contains no real credentials.
