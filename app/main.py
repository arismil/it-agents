from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes.tickets import router as tickets_router
from app.db import init_db

app = FastAPI(title="IT Helpdesk Agent")
init_db()


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = Path("app/templates/index.html").read_text()
    return HTMLResponse(content=html)


app.include_router(tickets_router)
