from datetime import UTC, datetime

from app.db import get_connection


class TicketRepository:
    def save_ticket(self, payload: dict) -> dict:
        now = datetime.now(UTC).isoformat()
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO tickets (
                    ticket_id,
                    title,
                    description,
                    severity,
                    contact,
                    status,
                    priority,
                    summary,
                    next_action,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["ticket_id"],
                    payload["title"],
                    payload["description"],
                    payload["severity"],
                    payload["contact"],
                    payload["status"],
                    payload["priority"],
                    payload["summary"],
                    payload["next_action"],
                    now,
                ),
            )
            conn.commit()
            payload["id"] = cursor.lastrowid
        return payload

    def get_ticket(self, ticket_id: str) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)
