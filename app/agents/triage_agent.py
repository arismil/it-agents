class TriageAgent:
    """Determines severity and queue placement for a ticket."""

    def prioritize(self, severity: str, category: str) -> str:
        severity_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
        }
        return severity_map.get(severity.lower(), "medium")
