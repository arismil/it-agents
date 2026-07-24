from uuid import uuid4

from app.agents.gemini_adapter import GeminiAdapter
from app.agents.intake_agent import IntakeAgent
from app.agents.resolution_agent import ResolutionAgent
from app.agents.triage_agent import TriageAgent
from app.repositories.ticket_repository import TicketRepository


class TicketService:
    def __init__(
        self,
        gemini_adapter: GeminiAdapter | None = None,
        intake_agent: IntakeAgent | None = None,
        triage_agent: TriageAgent | None = None,
        resolution_agent: ResolutionAgent | None = None,
        repository: TicketRepository | None = None,
    ):
        self.gemini_adapter = gemini_adapter or GeminiAdapter()
        self.intake_agent = intake_agent or IntakeAgent()
        self.triage_agent = triage_agent or TriageAgent()
        self.resolution_agent = resolution_agent or ResolutionAgent()
        self.repository = repository or TicketRepository()

    def submit_ticket(self, title: str, description: str, severity: str, contact: str):
        classification = self.intake_agent.classify(title, description)
        priority = self.triage_agent.prioritize(severity, classification["category"])
        triage_text = self.gemini_adapter.triage(title, description, severity)
        next_action = self.resolution_agent.build_resolution(
            classification["category"], priority, contact
        )

        payload = {
            "ticket_id": f"TKT-{uuid4().hex[:8].upper()}",
            "title": title,
            "description": description,
            "severity": severity,
            "contact": contact,
            "status": "received",
            "summary": f"{title}: {triage_text[:160]}",
            "priority": priority,
            "next_action": next_action,
        }
        self.repository.save_ticket(payload)
        return payload

    def get_ticket(self, ticket_id: str):
        return self.repository.get_ticket(ticket_id)
