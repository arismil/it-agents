"""Controlled simulated enterprise tools; no production systems are contacted."""

from typing import Any


def search_logs(service: str, error: str | None = None) -> dict[str, Any]:
    if service == "payment-service":
        return {"service": service, "matches": ["connection pool exhausted", "timeout acquiring connection"]}
    if service == "identity-service":
        return {"service": service, "matches": ["token signature mismatch", "invalid signing key"]}
    if service == "order-service":
        return {"service": service, "matches": ["slow query on orders table", "query plan changed"]}
    return {"service": service, "matches": [error or "no matching error signature"]}


def get_service_metrics(service: str) -> dict[str, Any]:
    defaults = {
        "payment-service": {"error_rate": 0.18, "db_connections": 1.0},
        "identity-service": {"login_failures": 0.42, "idp_latency_ms": 120},
        "order-service": {"p95_latency_ms": 4800, "db_query_latency_ms": 2100, "cpu": 0.46},
    }
    return defaults.get(service, {"error_rate": 0.05, "health": "degraded"})


def search_knowledge_base(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "recommendation": (
            "Increase the connection pool and recycle the payment service"
            if "payment" in query or "database" in query
            else "Validate the service configuration and roll back the latest change"
        ),
    }


def get_incident_history(service: str) -> dict[str, Any]:
    return {"service": service, "similar_incidents": 2 if service in {"payment-service", "order-service"} else 1}


def restart_service(service: str) -> dict[str, Any]:
    return {"service": service, "action": "restart", "simulated": True, "accepted": True}


def check_service_health(service: str, attempt: int = 1) -> dict[str, Any]:
    # The first check fails for payment incidents to exercise replanning.
    healthy = not (service == "payment-service" and attempt == 1)
    return {"service": service, "healthy": healthy, "latency_ms": 180 if healthy else 5000}
