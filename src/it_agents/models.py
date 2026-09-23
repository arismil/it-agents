"""Pydantic schemas for requests, agent outputs and the executive report."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["security", "procurement_finance", "legal_compliance", "ai_governance"]
ALL_DOMAINS: list[str] = ["security", "procurement_finance", "legal_compliance", "ai_governance"]

EvidenceType = Literal["retrieved_evidence", "ai_inference", "missing_evidence"]
Severity = Literal["info", "low", "medium", "high", "critical"]
Compliance = Literal["compliant", "non_compliant", "partial", "unknown"]
RiskLevel = Literal["Low", "Moderate", "High", "Critical"]
Recommendation = Literal["Approve", "Conditional", "Reject"]
DataClass = Literal["Public", "Internal", "Confidential", "Highly Confidential"]


# --------------------------------------------------------------------------- request
class AssessmentRequest(BaseModel):
    vendor_name: str = Field(min_length=2, max_length=120)
    use_case: str = Field(min_length=10, max_length=1500)
    data_classification: DataClass = "Confidential"
    annual_spend_usd: float | None = Field(default=None, ge=0)
    handles_customer_pii: bool = False
    requested_by: str = Field(default="procurement@northstarfin.com", max_length=120)
    # Dev/test only: names of MCP tools or specialist agents to force-fail (resilience demo).
    simulate_failures: list[str] = Field(default_factory=list)


# ------------------------------------------------------------- specialist (LLM) output
# LLM-facing schemas keep every field required (no defaults) for strict structured output.
class Citation(BaseModel):
    source_id: str = Field(description="Exact chunk_id from a search result (e.g. 'POL-SEC-2025-04:p2:c1') or 'tool:<tool_name>' for MCP tool data.")
    quote: str = Field(description="Short verbatim excerpt (max ~30 words) copied exactly from the cited chunk/tool output.")


class FindingDraft(BaseModel):
    """One requirement's finding as written by a specialist (veto status is decided in code, not by the LLM)."""
    title: str
    detail: str = Field(description="What was found and why it matters, referencing the NFS policy requirement.")
    evidence_type: EvidenceType = Field(description="retrieved_evidence = directly supported by a cited vendor/policy chunk; ai_inference = your reasoning beyond what documents state; missing_evidence = required evidence not found.")
    severity: Severity
    compliance_status: Compliance
    policy_reference: str = Field(description="NFS policy id and section, e.g. 'POL-PROC-2025-02 Sec. 5'. Empty string if none.")
    citations: list[Citation]


class Finding(FindingDraft):
    requirement_id: str = ""
    veto_trigger: bool = False  # set deterministically: veto requirement + non_compliant + verified evidence


class DomainAssessment(BaseModel):
    domain: Domain
    summary: str
    risk_score: int
    findings: list[Finding]
    missing_evidence: list[str]
    contradictions: list[str]
    remediation: list[str]
    injection_attempts_observed: list[str]


class OrchestratorSynthesis(BaseModel):
    plan_steps_completed: list[str]
    vendor_tier: str = Field(description="NFS procurement tier with justification, e.g. 'Tier 1 (Critical) - spend > $250k and AI platform'.")
    executive_summary: str
    recommendation: Recommendation
    risk_score: int = Field(description="Composite Vendor Risk Score 0-100.")
    rationale: str
    cross_domain_contradictions: list[str]
    required_remediation: list[str]


# -------------------------------------------------------------------------- report
class CitationCheck(BaseModel):
    domain: str
    finding: str
    source_id: str
    quote: str
    valid: bool
    reason: str
    document: str | None = None
    page: int | None = None


class GuardrailEvent(BaseModel):
    stage: Literal["input", "tool", "retrieval", "output", "policy"]
    type: str
    detail: str
    action: str
    at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MissingEvidenceItem(BaseModel):
    domain: str
    item: str


class HumanReview(BaseModel):
    required: bool = True
    reasons: list[str] = Field(default_factory=list)
    status: Literal["pending", "approved", "overridden", "rejected", "not_required"] = "pending"
    reviewer: str | None = None
    final_recommendation: Recommendation | None = None
    comments: str | None = None
    decided_at: str | None = None


class AssessmentReport(BaseModel):
    assessment_id: str
    status: Literal["pending_human_review", "finalized", "blocked", "failed"]
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request: AssessmentRequest
    vendor_tier: str = ""
    recommendation: Recommendation | None = None  # None until the assessment has run (e.g. blocked requests)
    risk_rating: RiskLevel | None = None
    risk_score: int | None = None
    executive_summary: str = ""
    rationale: str = ""
    domain_findings: list[DomainAssessment] = Field(default_factory=list)
    incomplete_domains: list[str] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    cross_domain_contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[MissingEvidenceItem] = Field(default_factory=list)
    required_remediation: list[str] = Field(default_factory=list)
    citations: list[CitationCheck] = Field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)
    plan_steps_completed: list[str] = Field(default_factory=list)
    human_review: HumanReview = Field(default_factory=HumanReview)
    metrics: dict = Field(default_factory=dict)
