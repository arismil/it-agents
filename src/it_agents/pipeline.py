"""End-to-end assessment pipeline (LangGraph):

intake/input guardrails -> deep agent (plan + specialist subagents via MCP/RAG) -> output guardrails &
policy enforcement -> human review (interrupt) -> finalize
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import rag
from .agents import SPECIALISTS, build_orchestrator, to_domain_assessment
from .config import AGENT_RECURSION_LIMIT, DATA_DIR
from .guardrails import (
    InputRejected,
    enforce_policy,
    mask_pii,
    scan_output,
    validate_citations,
    validate_request,
    vendor_tier,
)
from .models import (
    ALL_DOMAINS,
    AssessmentReport,
    AssessmentRequest,
    DomainAssessment,
    GuardrailEvent,
    HumanReview,
    MissingEvidenceItem,
    OrchestratorSynthesis,
)
from .observability import RunTracer, audit, langfuse_handler

log = logging.getLogger(__name__)
REPORTS_DIR = DATA_DIR / "assessments"
CANARIES = ["CANARY-HELIOS-7731"]


class PipelineState(TypedDict, total=False):
    assessment_id: str
    request: dict
    report: dict
    agent_output: dict


def _events(report: AssessmentReport, new: list[GuardrailEvent]) -> None:
    report.guardrail_events.extend(new)


# ------------------------------------------------------------------------ nodes
async def intake(state: PipelineState) -> dict:
    req = AssessmentRequest.model_validate(state["request"])
    report = AssessmentReport(assessment_id=state["assessment_id"], status="pending_human_review", request=req)
    audit(report.assessment_id, "request_received", vendor=req.vendor_name, requested_by=req.requested_by)
    try:
        _events(report, validate_request(req))
    except InputRejected as exc:
        _events(report, exc.events)
        report.status = "blocked"
        report.executive_summary = "Request blocked by input guardrails: " + str(exc)
        report.human_review = HumanReview(required=False, status="not_required")
        audit(report.assessment_id, "request_blocked", reason=str(exc))
        return {"report": report.model_dump()}
    report.vendor_tier = vendor_tier(req)
    return {"report": report.model_dump()}


def _mcp_connection(simulated_tool_failures: list[str]) -> dict:
    if url := os.getenv("MCP_SERVER_URL"):
        return {"url": url, "transport": "streamable_http"}
    env = {**os.environ, "NFS_SIMULATE_FAILURES": ",".join(simulated_tool_failures), "ANONYMIZED_TELEMETRY": "False"}
    return {"command": sys.executable, "args": ["-m", "it_agents.mcp_server"], "transport": "stdio", "env": env}


def _collect_domain_results(messages: list) -> tuple[dict[str, DomainAssessment], dict[str, str]]:
    """Map task tool calls -> specialist structured outputs (latest wins if a domain ran twice)."""
    call_domain: dict[str, str] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                if tc["name"] == "task":
                    call_domain[tc["id"]] = tc["args"].get("subagent_type", "")
    results, failures = {}, {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id in call_domain:
            domain = call_domain[m.tool_call_id]
            if domain not in SPECIALISTS:
                continue
            try:
                results[domain] = to_domain_assessment(domain, m.content)
                failures.pop(domain, None)
            except Exception:
                if domain not in results:
                    failures[domain] = str(m.content)[:300]
    return results, failures


async def run_agents(state: PipelineState) -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    report = AssessmentReport.model_validate(state["report"])
    if report.status == "blocked":
        return {}
    req = report.request
    fail_agents = {f for f in req.simulate_failures if f in SPECIALISTS}
    fail_tools = [f for f in req.simulate_failures if f not in SPECIALISTS]
    tracer = RunTracer(report.assessment_id)
    callbacks: list[Any] = [tracer]
    if lf := langfuse_handler():
        callbacks.append(lf)
    events: list[GuardrailEvent] = []

    task = {
        "assessment_id": report.assessment_id,
        "vendor_name": req.vendor_name,
        "use_case": mask_pii(req.use_case),
        "data_classification": req.data_classification,
        "annual_spend_usd": req.annual_spend_usd,
        "handles_customer_pii": req.handles_customer_pii,
        "procurement_tier": report.vendor_tier,
        "required_domains": ALL_DOMAINS,
    }
    output: dict[str, Any] = {"domains": {}, "failures": {}, "synthesis": None, "todos": [], "error": None}
    try:
        client = MultiServerMCPClient({"nfs-enterprise": _mcp_connection(fail_tools)})
        async with client.session("nfs-enterprise") as session:
            tools = await load_mcp_tools(session)
            tracer.event("mcp_connected", tools=[t.name for t in tools])
            agent = build_orchestrator(tools, events, fail_agents)
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Perform the vendor risk assessment for this structured "
                                                          "request:\n" + json.dumps(task, indent=2)}]},
                config={"recursion_limit": AGENT_RECURSION_LIMIT, "callbacks": callbacks,
                        "run_name": f"vendor-assessment:{req.vendor_name}",
                        "metadata": {"assessment_id": report.assessment_id}},
            )
        domains, failures = _collect_domain_results(result["messages"])
        synthesis = result.get("structured_response")
        output.update(
            domains={k: v.model_dump() for k, v in domains.items()},
            failures=failures,
            synthesis=synthesis.model_dump() if isinstance(synthesis, OrchestratorSynthesis) else None,
            todos=result.get("todos", []),
        )
    except Exception as exc:  # graceful degradation: produce a report that says what failed
        log.exception("agent run failed")
        output["error"] = f"{type(exc).__name__}: {exc}"[:500]
        events.append(GuardrailEvent(stage="tool", type="agent_run_failure", detail=output["error"], action="degraded"))

    if lf:  # export traces now - the CLI process may exit before the background exporter runs
        from langfuse import get_client

        get_client().flush()
    report.metrics = tracer.metrics()
    report.guardrail_events.extend(events)
    tracer.save({"todos": output["todos"], "tool_outputs": tracer.tool_outputs})
    output["tool_outputs"] = tracer.tool_outputs
    audit(report.assessment_id, "agents_completed", metrics=report.metrics, error=output["error"])
    return {"report": report.model_dump(), "agent_output": output}


async def enforce(state: PipelineState) -> dict:
    report = AssessmentReport.model_validate(state["report"])
    if report.status == "blocked":
        return {}
    out = state["agent_output"]
    domains = [DomainAssessment.model_validate(d) for d in out["domains"].values()]
    incomplete = [d for d in ALL_DOMAINS if d not in out["domains"]]
    record = rag.resolve_vendor(report.request.vendor_name)
    aliases = {report.request.vendor_name, *(([record["vendor_name"], *record["aliases"]]) if record else [])}

    # 1. citation verification (relabels unsupported claims)
    checks, cite_events = validate_citations(domains, aliases, out.get("tool_outputs", {}))
    report.citations = checks
    _events(report, cite_events)

    # 2. evidence gaps
    missing = [MissingEvidenceItem(domain=d.domain, item=i) for d in domains for i in d.missing_evidence]
    missing += [MissingEvidenceItem(domain=d, item=f"Domain assessment incomplete: specialist unavailable "
                                                   f"({out['failures'].get(d, 'no result returned')[:120]})")
                for d in incomplete]
    flagged = sum(o.count('"injection_flagged": true') for o in out.get("tool_outputs", {}).values())
    if flagged:
        _events(report, [GuardrailEvent(stage="retrieval", type="injection_redacted_at_mcp",
                                        detail=f"{flagged} retrieved chunk(s) contained prompt-injection text; "
                                               "offending sentences were redacted before reaching the agents",
                                        action="redacted")])
    for d in domains:
        for inj in d.injection_attempts_observed:
            _events(report, [GuardrailEvent(stage="retrieval", type="injection_reported_by_agent",
                                            detail=f"{d.domain}: {inj}"[:400], action="ignored")])

    # 3. policy enforcement on the recommendation
    synth = OrchestratorSynthesis.model_validate(out["synthesis"]) if out.get("synthesis") else None
    llm_rec = synth.recommendation if synth else "Reject"
    rec, rating, score, violations, pol_events = enforce_policy(llm_rec, domains, incomplete, len(missing))
    _events(report, pol_events)

    report.domain_findings = domains
    report.incomplete_domains = incomplete
    report.missing_evidence = missing
    report.policy_violations = violations
    report.recommendation, report.risk_rating, report.risk_score = rec, rating, score
    if synth:
        report.executive_summary = synth.executive_summary
        report.rationale = synth.rationale
        report.cross_domain_contradictions = synth.cross_domain_contradictions + [
            c for d in domains for c in d.contradictions]
        report.required_remediation = synth.required_remediation or [r for d in domains for r in d.remediation]
        report.plan_steps_completed = synth.plan_steps_completed
        report.vendor_tier = synth.vendor_tier or report.vendor_tier
    else:
        report.status = "failed" if out.get("error") else report.status
        report.executive_summary = ("Automated synthesis unavailable" + (f" ({out['error']})" if out.get("error") else "")
                                    + ". Findings below are partial; a manual assessment is required.")
        report.required_remediation = [r for d in domains for r in d.remediation]

    # 4. output scanning (injection echo / canary leakage) + PII masking
    text = report.model_dump_json(include={"executive_summary", "rationale", "required_remediation"})
    _events(report, scan_output(text, CANARIES))
    report.executive_summary = mask_pii(report.executive_summary)
    report.rationale = mask_pii(report.rationale)

    # 5. human-in-the-loop requirements
    reasons = ["Final vendor approval/rejection is a Tier C decision requiring human sign-off (POL-AI-2025-01 §4)"]
    if rating in ("High", "Critical"):
        reasons.append(f"{rating} risk rating")
    if violations:
        reasons.append(f"{len(violations)} policy violation(s) / veto trigger(s)")
    if incomplete:
        reasons.append(f"Incomplete domains: {', '.join(incomplete)}")
    if any(e.type in ("recommendation_override", "canary_leak", "injection_echo") for e in report.guardrail_events):
        reasons.append("Guardrail intervention on the AI output")
    if any(e.stage == "retrieval" for e in report.guardrail_events):
        reasons.append("Prompt-injection attempt detected in vendor evidence")
    report.human_review = HumanReview(required=True, reasons=reasons, status="pending")
    if report.status != "failed":
        report.status = "pending_human_review"
    audit(report.assessment_id, "report_generated", recommendation=rec, risk=rating, score=score)
    return {"report": report.model_dump()}


async def human_review(state: PipelineState) -> dict:
    report = AssessmentReport.model_validate(state["report"])
    if report.status == "blocked":
        return {}
    decision = interrupt({"assessment_id": report.assessment_id, "recommendation": report.recommendation,
                          "risk_rating": report.risk_rating, "reasons": report.human_review.reasons})
    action = decision.get("decision", "accept")
    hr = report.human_review
    hr.reviewer = decision.get("reviewer", "unknown")
    hr.comments = decision.get("comments")
    hr.decided_at = datetime.now(timezone.utc).isoformat()
    if action == "accept":
        hr.status, hr.final_recommendation = "approved", report.recommendation
    elif action == "override":
        hr.status, hr.final_recommendation = "overridden", decision.get("final_recommendation") or report.recommendation
    else:  # return_for_rework
        hr.status, hr.final_recommendation = "rejected", None
    report.status = "finalized"
    audit(report.assessment_id, "human_decision", decision=action, reviewer=hr.reviewer,
          final_recommendation=hr.final_recommendation)
    return {"report": report.model_dump()}


def _route_after_intake(state: PipelineState) -> str:
    return END if state["report"]["status"] == "blocked" else "run_agents"


def build_graph(checkpointer=None):
    g = StateGraph(PipelineState)
    g.add_node("intake", intake)
    g.add_node("run_agents", run_agents)
    g.add_node("enforce_guardrails", enforce)
    g.add_node("human_review", human_review)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", _route_after_intake, ["run_agents", END])
    g.add_edge("run_agents", "enforce_guardrails")
    g.add_edge("enforce_guardrails", "human_review")
    g.add_edge("human_review", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ------------------------------------------------------------------------ facade
class AssessmentService:
    def __init__(self):
        self.graph = build_graph()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save(self, report: dict) -> AssessmentReport:
        rep = AssessmentReport.model_validate(report)
        (REPORTS_DIR / f"{rep.assessment_id}.json").write_text(rep.model_dump_json(indent=2))
        return rep

    async def start(self, request: AssessmentRequest, assessment_id: str | None = None) -> AssessmentReport:
        aid = assessment_id or uuid.uuid4().hex[:12]
        state = await self.graph.ainvoke({"assessment_id": aid, "request": request.model_dump()},
                                         config={"configurable": {"thread_id": aid}})
        return self._save(state["report"])

    async def review(self, assessment_id: str, decision: dict) -> AssessmentReport:
        state = await self.graph.ainvoke(Command(resume=decision), config={"configurable": {"thread_id": assessment_id}})
        return self._save(state["report"])

    def awaiting_review(self, assessment_id: str) -> bool:
        snap = self.graph.get_state({"configurable": {"thread_id": assessment_id}})
        return bool(snap and snap.next)

    @staticmethod
    def load(assessment_id: str) -> AssessmentReport | None:
        p = REPORTS_DIR / f"{assessment_id}.json"
        return AssessmentReport.model_validate_json(p.read_text()) if p.exists() else None

    @staticmethod
    def list() -> list[dict]:
        out = []
        for p in sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            r = json.loads(p.read_text())
            out.append({k: r.get(k) for k in ("assessment_id", "status", "recommendation", "risk_rating",
                                              "generated_at")} | {"vendor": r["request"]["vendor_name"]})
        return out
