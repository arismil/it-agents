"""FastAPI application: REST API + single-page UI for assessments and human review."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from . import rag
from .config import DATA_DIR, PACKAGE_DIR
from .models import AssessmentRequest, Recommendation
from .observability import load_trace, setup_logging
from .pipeline import AssessmentService
from .report import to_markdown

log = logging.getLogger(__name__)
service: AssessmentService | None = None
running: dict[str, str] = {}  # assessment_id -> "running" | "error: ..."
_tasks: set[asyncio.Task] = set()

SAMPLES = [
    {"label": "Asteria AI Systems (target)", "request": {
        "vendor_name": "Asteria AI Systems", "data_classification": "Confidential",
        "use_case": "GenAI processing of confidential documents (contracts, internal audit reports) for NFS risk "
                    "and compliance analysts."}},
    {"label": "Vendor X / ApexCognition", "request": {
        "vendor_name": "Vendor X", "data_classification": "Confidential", "annual_spend_usd": 320000,
        "use_case": "GenAI platform (ApexCognition) to process and summarise confidential internal documents for "
                    "NFS risk & compliance teams."}},
    {"label": "Helios Doc AI (prompt-injection test)", "request": {
        "vendor_name": "Helios Doc AI", "data_classification": "Confidential", "annual_spend_usd": 120000,
        "use_case": "GenAI summarisation service for confidential NFS documents."}},
    {"label": "Vendor X with failures (resilience test)", "request": {
        "vendor_name": "Vendor X", "data_classification": "Confidential", "annual_spend_usd": 320000,
        "use_case": "GenAI platform to process confidential internal documents for NFS risk teams.",
        "simulate_failures": ["get_financial_risk_intel", "ai_governance"]}},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    setup_logging()
    if not rag.index_ready():
        log.info("RAG index missing - ingesting corpus")
        await asyncio.to_thread(rag.ingest)
    service = AssessmentService()
    yield


app = FastAPI(title="NFS Vendor Risk Deep Agent", version="0.1.0", lifespan=lifespan)


class ReviewDecision(BaseModel):
    decision: Literal["accept", "override", "return_for_rework"]
    reviewer: str
    final_recommendation: Recommendation | None = None
    comments: str | None = None


async def _run(aid: str, req: AssessmentRequest) -> None:
    try:
        await service.start(req, assessment_id=aid)
        running.pop(aid, None)
    except Exception as exc:  # surfaced to the UI instead of crashing the server
        log.exception("assessment %s failed", aid)
        running[aid] = f"error: {exc}"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PACKAGE_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "index_ready": rag.index_ready(), "running": len(running)}


@app.get("/api/samples")
def samples():
    return SAMPLES


@app.post("/api/ingest")
async def reingest():
    return await asyncio.to_thread(rag.ingest)


@app.post("/api/assessments", status_code=202)
async def create(req: AssessmentRequest):
    import uuid

    aid = uuid.uuid4().hex[:12]
    running[aid] = "running"
    task = asyncio.create_task(_run(aid, req))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"assessment_id": aid, "status": "running"}


@app.get("/api/assessments")
def list_assessments():
    active = [{"assessment_id": k, "status": v, "vendor": None} for k, v in running.items()]
    return active + AssessmentService.list()


@app.get("/api/assessments/{aid}")
def get_assessment(aid: str):
    if aid in running:
        return {"assessment_id": aid, "status": running[aid]}
    report = AssessmentService.load(aid)
    if not report:
        raise HTTPException(404, "assessment not found")
    return report


@app.get("/api/assessments/{aid}/markdown", response_class=PlainTextResponse)
def get_markdown(aid: str):
    report = AssessmentService.load(aid)
    if not report:
        raise HTTPException(404, "assessment not found")
    return to_markdown(report)


@app.post("/api/assessments/{aid}/review")
async def review(aid: str, decision: ReviewDecision):
    report = AssessmentService.load(aid)
    if not report:
        raise HTTPException(404, "assessment not found")
    if not service.awaiting_review(aid):
        raise HTTPException(409, f"assessment is not awaiting human review (status: {report.status})")
    if decision.decision == "override" and not decision.final_recommendation:
        raise HTTPException(422, "override requires final_recommendation")
    return await service.review(aid, decision.model_dump())


@app.get("/api/assessments/{aid}/trace")
def trace(aid: str):
    t = load_trace(aid)
    if not t:
        raise HTTPException(404, "trace not found")
    t.pop("tool_outputs", None)
    return t


@app.get("/api/evals/latest")
def latest_eval():
    p = DATA_DIR / "evals" / "latest.json"
    if not p.exists():
        raise HTTPException(404, "no evaluation run yet - run `it-agents eval`")
    return json.loads(p.read_text())
