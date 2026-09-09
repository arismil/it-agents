from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .models import Evidence, IncidentReport, IncidentRequest, IncidentStatus, Severity
from .observability import resolution_trace
from . import tools


class WorkflowState(TypedDict, total=False):
    request: IncidentRequest
    report: IncidentReport
    trace_id: str
    evidence: list[Evidence]
    approval: bool | None
    attempts: int
    error: str


def _report(state: WorkflowState) -> IncidentReport:
    return state["report"]


def triage(state: WorkflowState) -> dict[str, Any]:
    request = state["request"]
    service = request.service.lower()
    incident_type = (
        "payment" if "payment" in service else
        "authentication" if service in {"identity-service", "auth-service"} else
        "performance" if service == "order-service" else "application"
    )
    severity = request.severity or (Severity.high if incident_type in {"payment", "authentication"} else Severity.medium)
    report = IncidentReport(
        incident_id=request.incident_id, service=request.service,
        status=IncidentStatus.investigating, incident_type=incident_type, severity=severity,
        trace_id=state["trace_id"],
    )
    return {"report": report}


def investigate(state: WorkflowState) -> dict[str, Any]:
    request, report = state["request"], _report(state)
    evidence: list[Evidence] = []
    try:
        logs = tools.search_logs(request.service, request.error)
        metrics = tools.get_service_metrics(request.service)
        knowledge = tools.search_knowledge_base(f"{report.incident_type} {request.service} {request.description}")
        history = tools.get_incident_history(request.service)
        evidence.extend([
            Evidence(source="logs", finding="Diagnostic log matches collected", details=logs),
            Evidence(source="metrics", finding="Service metrics collected", details=metrics),
            Evidence(source="knowledge_base", finding="Operational guidance found", details=knowledge),
            Evidence(source="incident_history", finding="Similar incident history collected", details=history),
        ])
    except Exception as exc:
        return {"error": f"investigation failed: {exc}", "evidence": evidence}
    report.evidence = evidence
    return {"report": report, "evidence": evidence}


def diagnose(state: WorkflowState) -> dict[str, Any]:
    report = _report(state)
    causes = {
        "payment": "Payment database connection pool exhaustion",
        "authentication": "Identity token signing/configuration mismatch",
        "performance": "Order database query degradation",
        "application": "Application dependency or configuration degradation",
    }
    report.root_cause = causes.get(report.incident_type or "application", causes["application"])
    report.remediation = "Recycle the affected service and apply the known safe configuration fix"
    report.risk_level = "high" if report.severity in {Severity.high, Severity.critical} else "low"
    report.approval_required = report.risk_level == "high"
    report.approval_status = "pending" if report.approval_required else "not_required"
    report.status = IncidentStatus.awaiting_approval if report.approval_required else IncidentStatus.remediating
    return {"report": report}


def route_after_diagnosis(state: WorkflowState) -> str:
    return "approval" if _report(state).approval_required else "execute"


def approval(state: WorkflowState) -> dict[str, Any]:
    if state.get("approval") is False:
        report = _report(state)
        report.status, report.approval_status = IncidentStatus.failed, "rejected"
        report.resolution_summary = "Remediation was rejected by an operator."
    elif state.get("approval") is True:
        report = _report(state)
        report.status, report.approval_status = IncidentStatus.remediating, "approved"
    return {"report": _report(state)}


def route_after_approval(state: WorkflowState) -> str:
    return "execute" if state.get("approval") is True else "stop"


def execute(state: WorkflowState) -> dict[str, Any]:
    report = _report(state)
    result = tools.restart_service(report.service)
    attempt = state.get("attempts", 0) + 1
    report.attempts = attempt
    report.execution_result = f"Simulated {result['action']} accepted"
    report.status = IncidentStatus.remediating
    return {"report": report, "attempts": attempt}


def verify(state: WorkflowState) -> dict[str, Any]:
    report = _report(state)
    health = tools.check_service_health(report.service, state.get("attempts", 1))
    report.verification_result = "healthy" if health["healthy"] else "still unhealthy"
    if health["healthy"]:
        report.status = IncidentStatus.resolved
        report.resolution_summary = f"{report.root_cause}. Remediation verified successfully."
    return {"report": report}


def route_after_verify(state: WorkflowState) -> str:
    if _report(state).status == IncidentStatus.resolved:
        return "close"
    return "replan" if state.get("attempts", 0) < 3 else "fail"


def replan(state: WorkflowState) -> dict[str, Any]:
    report = _report(state)
    report.remediation = "Increase capacity/apply the alternate safe remediation, then restart the service"
    return {"report": report}


def fail(state: WorkflowState) -> dict[str, Any]:
    report = _report(state)
    report.status = IncidentStatus.failed
    report.resolution_summary = "Verification failed after the maximum remediation attempts."
    return {"report": report}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("triage", triage)
    graph.add_node("investigate", investigate)
    graph.add_node("diagnose", diagnose)
    graph.add_node("approval", approval)
    graph.add_node("execute", execute)
    graph.add_node("verify", verify)
    graph.add_node("replan", replan)
    graph.add_node("fail", fail)
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "investigate")
    graph.add_edge("investigate", "diagnose")
    graph.add_conditional_edges("diagnose", route_after_diagnosis, {"approval": "approval", "execute": "execute"})
    graph.add_conditional_edges("approval", route_after_approval, {"execute": "execute", "stop": END})
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"close": END, "replan": "replan", "fail": "fail"})
    graph.add_edge("replan", "execute")
    graph.add_edge("fail", END)
    return graph.compile()


workflow = build_graph()


def resolve_incident(request: IncidentRequest, approval: bool | None = None) -> IncidentReport:
    trace_id = str(uuid4())
    with resolution_trace(trace_id, request):
        result = workflow.invoke({
            "request": request, "trace_id": trace_id, "approval": approval, "attempts": 0,
        })
    report = result["report"]
    if result.get("error"):
        report.errors.append(result["error"])
        report.status = IncidentStatus.failed
    return report
