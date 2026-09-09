from it_agents.models import IncidentRequest, IncidentStatus, Severity
from it_agents.workflow import resolve_incident


def test_payment_incident_replans_after_failed_verification():
    report = resolve_incident(IncidentRequest(
        incident_id="INC-1042", service="payment-service",
        description="Customers report payment failures", error="Database connection timeout",
    ), approval=True)
    assert report.status == IncidentStatus.resolved
    assert report.attempts == 2
    assert report.root_cause == "Payment database connection pool exhaustion"
    assert len(report.evidence) >= 3


def test_high_risk_incident_waits_for_approval():
    report = resolve_incident(IncidentRequest(
        incident_id="INC-1", service="identity-service",
        description="Users cannot log in", severity=Severity.high,
    ))
    assert report.status == IncidentStatus.awaiting_approval
    assert report.approval_required is True


def test_low_risk_incident_executes_without_approval():
    report = resolve_incident(IncidentRequest(
        incident_id="INC-2", service="order-service",
        description="Latency increased", severity=Severity.low,
    ))
    assert report.status == IncidentStatus.resolved
    assert report.approval_status == "not_required"
