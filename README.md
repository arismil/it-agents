# IT Agents

A lightweight FastAPI scaffold for learning to build an agentic IT automation helpdesk with Gemini-style orchestration.

## Project goals

- Expose a small FastAPI API for helpdesk ticket submission
- Separate HTTP, agent, schema, and service concerns
- Keep the first version easy to understand and extend
- Provide a simple place to plug in a real Gemini provider later

## Structure

- `app/main.py` — FastAPI application entrypoint
- `app/api/routes/tickets.py` — API route definitions for health and ticket submission
- `app/agents/orchestrator.py` — simple orchestration layer
- `app/agents/gemini_adapter.py` — adapter boundary for model integration
- `app/services/ticket_service.py` — ticket handling logic
- `app/schemas/ticket.py` — request/response models

## Run locally

```bash
source .venv/bin/activate
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-2.0-flash"
uvicorn app.main:app --reload
```

## Example request

```bash
curl -X POST http://127.0.0.1:8000/tickets/submit \
  -H "Content-Type: application/json" \
  -d '{
    "title": "VPN login failure",
    "description": "User cannot connect to VPN after password reset.",
    "severity": "high",
    "contact": "alex@example.com"
  }'
```

## Verify

```bash
source .venv/bin/activate
pytest -q
```

## Next learning steps

1. Replace the placeholder Gemini adapter with a real Google Generative AI call
2. Add a persistence layer for tickets
3. Add a knowledge-base/tool action layer for troubleshooting workflows
4. Convert the orchestrator into a more explicit multi-step agent flow
