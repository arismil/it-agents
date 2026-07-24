class ResolutionAgent:
    """Drafts the next action for the support queue."""

    def build_resolution(self, category: str, priority: str, contact: str) -> str:
        return (
            f"Route to {category} queue with {priority} priority and send a response to {contact}."
        )
