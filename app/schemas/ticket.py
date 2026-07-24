from pydantic import BaseModel, Field


class TicketCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    description: str = Field(..., min_length=5)
    severity: str = Field(default="medium")
    contact: str = Field(..., min_length=3)


class TicketResponse(BaseModel):
    ticket_id: str
    title: str
    description: str
    severity: str
    contact: str
    status: str
    summary: str
    priority: str
    next_action: str
