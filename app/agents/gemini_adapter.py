import os

try:
    from google import genai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    genai = None


class GeminiAdapter:
    """Adapter for Gemini-backed triage and response generation."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.client = None
        if genai is not None and self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def triage(self, title: str, description: str, severity: str) -> str:
        if self.client is None:
            return (
                f"Triage request: {title}. Description: {description}. "
                f"Severity is {severity}. Suggest a first-pass helpdesk response."
            )

        prompt = (
            "You are an IT support triage assistant. "
            f"Title: {title}\nDescription: {description}\nSeverity: {severity}\n"
            "Return a concise first-pass troubleshooting plan and next action."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        return text.strip() if text else str(response)
