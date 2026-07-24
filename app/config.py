import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "it-helpdesk-agent"
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
