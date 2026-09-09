from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .models import ApprovalRequest, IncidentReport, IncidentRequest, IncidentStatus
from .workflow import resolve_incident

app = FastAPI(title="AI IT Incident Resolution Agent", version="0.1.0")
reports: dict[str, IncidentReport] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/incidents", response_model=IncidentReport, status_code=201)
def create_incident(request: IncidentRequest) -> IncidentReport:
    report = resolve_incident(request)
    reports[request.incident_id] = report
    return report


@app.get("/incidents/{incident_id}", response_model=IncidentReport)
def get_incident(incident_id: str) -> IncidentReport:
    if incident_id not in reports:
        raise HTTPException(status_code=404, detail="Incident not found")
    return reports[incident_id]


@app.post("/incidents/{incident_id}/approval", response_model=IncidentReport)
def approve_incident(incident_id: str, decision: ApprovalRequest) -> IncidentReport:
    existing = reports.get(incident_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if existing.status != IncidentStatus.awaiting_approval:
        raise HTTPException(status_code=409, detail="Incident is not awaiting approval")
    request = IncidentRequest(
        incident_id=existing.incident_id, service=existing.service,
        description=existing.root_cause or "Operator-approved remediation",
        severity=existing.severity,
    )
    report = resolve_incident(request, approval=decision.approved)
    reports[incident_id] = report
    return report


def run() -> None:
    import uvicorn
    uvicorn.run("it_agents.api:app", host="0.0.0.0", port=8000, reload=False)
