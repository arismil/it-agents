from app.agents.gemini_adapter import GeminiAdapter


class TicketOrchestrator:
    """Simple agentic workflow for IT helpdesk intake and triage."""

    def __init__(self, gemini_adapter: GeminiAdapter | None = None):
        self.gemini_adapter = gemini_adapter or GeminiAdapter()

    def plan_response(self, title: str, description: str, severity: str) -> dict:
        triage = self.gemini_adapter.triage(title, description, severity)
        return {
            "goal": "acknowledge and route",
            "summary": triage,
            "priority": severity.lower(),
        }
