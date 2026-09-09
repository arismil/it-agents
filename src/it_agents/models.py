from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class IncidentStatus(StrEnum):
    investigating = "investigating"
    awaiting_approval = "awaiting_approval"
    remediating = "remediating"
    resolved = "resolved"
    failed = "failed"


class IncidentRequest(BaseModel):
    incident_id: str = Field(min_length=1)
    service: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Severity | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    source: str
    finding: str
    details: dict[str, Any] = Field(default_factory=dict)


class IncidentReport(BaseModel):
    incident_id: str
    service: str
    status: IncidentStatus
    incident_type: str | None = None
    severity: Severity | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    root_cause: str | None = None
    remediation: str | None = None
    risk_level: str | None = None
    approval_required: bool = False
    approval_status: str = "not_required"
    execution_result: str | None = None
    verification_result: str | None = None
    attempts: int = 0
    resolution_summary: str | None = None
    trace_id: str | None = None
    errors: list[str] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    approved: bool
