class IntakeAgent:
    """Detects intent and route category from a user request."""

    def classify(self, title: str, description: str) -> dict:
        lowered = f"{title} {description}".lower()
        if any(term in lowered for term in ["vpn", "access", "login", "password"]):
            category = "identity_access"
        elif any(term in lowered for term in ["printer", "network", "wifi", "connectivity"]):
            category = "connectivity"
        elif any(term in lowered for term in ["email", "outlook", "mail"]):
            category = "collaboration"
        else:
            category = "general_support"

        return {
            "category": category,
            "intent": "triage",
        }
