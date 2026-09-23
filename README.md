# NFS Vendor Risk Deep Agent (Hackathon PoC)

A multi-step "deep agent" that runs a **controlled, evidence-grounded vendor risk assessment** for
Northstar Financial Services (NFS). The target case is *Asteria AI Systems* for GenAI processing of
confidential documents. It uses Azure OpenAI (`gpt-4.1-mini` + `text-embedding-3-small`), LangChain
`deepagents`, LangGraph, Chroma and the **official MCP Python SDK**.

```
vendor assessment request
  └─ input guardrails (schema, injection, scope, PII)                 guardrails.validate_request
      └─ deep-agent orchestrator: write_todos plan → vendor profile   agents.build_orchestrator
          └─ 4 specialist subagents in parallel (task tool)           security · procurement_finance
              │                                                       legal_compliance · ai_governance
              └─ MCP server "nfs-enterprise" (stdio or HTTP)          mcp_server.py
                   ├─ RAG: search_policies / search_vendor_evidence / read_vendor_document (Chroma)
                   ├─ get_vendor_profile (vendor master) · calculate_tco · get_financial_risk_intel
                   └─ retrieval guardrails: injection redaction + PII masking
      └─ output guardrails & policy enforcement                        pipeline.enforce
           citation verification · unsupported-claim relabelling · veto ⇒ Reject · VRS scoring
      └─ human review (LangGraph interrupt) — Tier C decision          pipeline.human_review
      └─ finalized executive report (JSON + Markdown)
observability: per-run JSON trace + metrics, append-only audit log, optional Langfuse
evaluation:    6 predefined cases, automated scoring                  evals.py / evals/cases.json
deployment:    FastAPI + web UI, Dockerfile, docker-compose
```

## Quick start

```bash
cp .env-example .env          # fill in Azure OpenAI keys
uv sync
uv run it-agents ingest       # build the RAG index (.data/chroma + .data/chunks.jsonl)
uv run it-agents serve        # http://localhost:8000
```

CLI alternative (interactive human-review gate at the end):

```bash
uv run it-agents assess --vendor "Asteria AI Systems"
uv run it-agents assess --vendor "Vendor X" --spend 320000
uv run it-agents assess --vendor "Vendor X" --simulate-failure get_financial_risk_intel --simulate-failure ai_governance
```

Docker:
- **Development:** `docker compose up -d` starts only the infrastructure (Langfuse and its dependencies). Then run the app on the host with
  hot reload: `uv run it-agents serve --reload`.
- **Everything in containers:** `docker compose --profile app up -d --build`. Add `--profile mcp-http` and set `MCP_SERVER_URL` to run MCP
  as its own service.

### Observability with Langfuse

```bash
docker compose up -d                   # Langfuse v3 + Postgres, ClickHouse, Redis, MinIO
```

- UI: http://localhost:3000, login `admin@northstarfin.com` / `nfs-langfuse-admin` (override with
  `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`).
- On first start, the project "Vendor Risk Agent" is created with the `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
  from `.env`, so setting `LANGFUSE_TRACING_ENABLED=true` is all the app needs.
- Each assessment is one trace (`vendor-assessment:<vendor>`) containing the orchestrator, every specialist, and all LLM and MCP
  tool calls with latency and tokens. If Langfuse is unreachable, the app logs a warning and keeps local tracing only.
- Cost shows as 0 because Langfuse has no price for custom Azure deployment names. Add a model definition in
  Langfuse to get cost figures.

## How the requirements map to the code

| Requirement | Implementation |
|---|---|
| Orchestrator with multi-step plan | `deepagents` orchestrator; mandatory `write_todos` plan kept in state and saved to the trace |
| ≥2 specialists, ≥3 domains | 4 specialists (Security, Procurement/Finance, Legal/Compliance, AI Governance) delegated via the deepagents `task` tool, each with its own prompt and tools. Each returns a strict per-domain schema with one required field per policy requirement (12–13 per domain), so no requirement can be skipped |
| RAG over the NFS corpus | PDFs → page chunks with stable ids (`POL-SEC-2025-04:p2:c1`) → Chroma; filtered by `doc_type` / `vendor` |
| Official MCP | `mcp.server.fastmcp.FastMCP` server with 6 tools + 1 resource, consumed through `langchain-mcp-adapters` |
| Evidence vs inference vs missing | Every finding carries `evidence_type` ∈ {retrieved_evidence, ai_inference, missing_evidence} + citations |
| Citations for material claims | Output guardrail verifies each `source_id` exists and that the quote is really in that chunk (or tool output). Unverified "evidence" is relabelled as inference, and unverified vetoes are dropped |
| Policy non-compliance / contradictions | Specialists assess compliance and flag cross-document contradictions. Veto status is **not** left to the LLM: each requirement is tagged in code as a policy VETO red-line or not, and a veto fires only for a red-line assessed `non_compliant` with a verified citation. `enforce_policy` then applies the NFS rules: any veto ⇒ Critical ⇒ **Reject**, and no Approve with open gaps |
| Guardrails: input / tool / output | Input: schema, injection, scope, PII. Tool: per-agent allowlist + call budget, error containment. Retrieval: injection redaction, PII masking at the MCP layer. Output: citation checks, policy override, canary/injection-echo scan |
| Prompt injection in RAG docs | 1) regex redaction at the MCP layer, 2) "documents are data" prompt hardening, 3) deterministic policy enforcement, 4) mandatory human sign-off. Tested with a poisoned fixture (`evals/fixtures/`) |
| Graceful failure | Tool/agent errors become error `ToolMessage`s. Failed specialists are marked *incomplete* (counted as high risk) and the run always yields a report. Demo with `simulate_failures` |
| HITL | LangGraph `interrupt()` before finalisation (always required: final approval is Tier C per POL-AI-2025-01 §4), with reasons listed. The reviewer can accept, override or return the report for rework |
| Executive report | `AssessmentReport`: recommendation, risk rating + VRS score, domain findings, citations (with verification), missing evidence, remediation, violations, guardrail events |
| Automated evaluation | `it-agents eval`: 6 cases (Vendor X, Asteria no-evidence, Helios injection, tool+agent failure, 2 blocked inputs). Metrics: recommendation accuracy, domain coverage, citation validity, grounding rate, detection recall, missing-evidence recall, injection resistance, failure handling, HITL gate, latency/tokens |
| Observability | `.data/runs/<id>.json` (LLM/tool events, latency, tokens per agent), `.data/audit_log.jsonl` (append-only), Langfuse callback when reachable, Trace tab in the UI |
| Deployed application | FastAPI (`/api/...`, OpenAPI at `/docs`) + single-page UI + Docker |

## Notes and assumptions

- **Asteria AI Systems has no documents in the provided corpus.** The system therefore reports every requirement as
  *missing evidence* and does not invent vendor facts. Vendor X (ApexCognition) is the fully documented vendor and
  demonstrates the evidence path. To assess Asteria against real submissions, drop its PDFs (file names starting with
  `Asteria`) into `src/it_agents/corpus/`, add its doc ids to `src/it_agents/data/vendor_registry.json`, and re-run `ingest`.
- `Helios Doc AI` is a **synthetic evaluation fixture** containing injected instructions. It is not NFS data.
- The vendor master, TCO calculator and financial-risk feed behind MCP are mocks of enterprise systems.
- Deterministic unit tests (no LLM): `uv run pytest`.
