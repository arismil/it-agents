from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI(title="IT Helpdesk Demo")


@app.get("/", response_class=HTMLResponse)
def demo_page():
    html = Path("app/templates/index.html").read_text()
    return HTMLResponse(content=html)
