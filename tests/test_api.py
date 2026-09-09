from fastapi.testclient import TestClient

from it_agents.api import app, reports

client = TestClient(app)


def test_health_and_incident_lifecycle():
    reports.clear()
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/incidents", json={
        "incident_id": "INC-API", "service": "identity-service",
        "description": "Token validation errors",
    })
    assert response.status_code == 201
    assert response.json()["status"] == "awaiting_approval"
    approval = client.post("/incidents/INC-API/approval", json={"approved": True})
    assert approval.status_code == 200
    assert approval.json()["status"] == "resolved"
