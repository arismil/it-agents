from fastapi import APIRouter, HTTPException

from app.schemas.ticket import TicketCreate, TicketResponse
from app.services.ticket_service import TicketService

router = APIRouter()
service = TicketService()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/tickets/submit", response_model=TicketResponse)
def submit_ticket(payload: TicketCreate) -> TicketResponse:
    result = service.submit_ticket(
        title=payload.title,
        description=payload.description,
        severity=payload.severity,
        contact=payload.contact,
    )
    return TicketResponse(**result)


@router.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: str) -> TicketResponse:
    result = service.get_ticket(ticket_id)
    if result is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return TicketResponse(**result)
