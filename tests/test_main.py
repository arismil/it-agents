import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["IT_AGENTS_DB_PATH"] = str(Path("data/test-helpdesk.db"))

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ticket_submission_endpoint():
    response = client.post(
        "/tickets/submit",
        json={
            "title": "VPN login failure",
            "description": "User cannot connect to VPN after password reset.",
            "severity": "high",
            "contact": "alex@example.com",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "received"
    assert payload["summary"]
    assert payload["priority"] in {"low", "medium", "high", "critical"}


def test_ticket_is_persisted_and_retrievable():
    response = client.post(
        "/tickets/submit",
        json={
            "title": "Printer offline",
            "description": "User cannot print from finance department.",
            "severity": "medium",
            "contact": "sam@example.com",
        },
    )
    payload = response.json()

    lookup = client.get(f"/tickets/{payload['ticket_id']}")
    assert lookup.status_code == 200
    saved_ticket = lookup.json()
    assert saved_ticket["ticket_id"] == payload["ticket_id"]
    assert saved_ticket["title"] == "Printer offline"
