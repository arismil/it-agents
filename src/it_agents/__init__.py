"""NFS Vendor Risk Deep Agent - CLI entry point.

  it-agents ingest                      build the RAG index from the corpus
  it-agents assess --vendor "Vendor X"  run an assessment (interactive human review)
  it-agents serve                       start the web app / API
  it-agents eval [--case ID]            run the automated evaluation suite
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def _assess(args) -> None:
    from .models import AssessmentRequest
    from .pipeline import AssessmentService
    from .rag import index_ready, ingest
    from .report import to_markdown

    if not index_ready():
        ingest()
    req = AssessmentRequest(
        vendor_name=args.vendor, use_case=args.use_case, data_classification=args.classification,
        annual_spend_usd=args.spend, handles_customer_pii=args.pii,
        simulate_failures=[s for s in (args.simulate_failure or []) if s],
    )

    async def run():
        svc = AssessmentService()
        report = await svc.start(req)
        print(to_markdown(report))
        if report.status == "blocked" or not svc.awaiting_review(report.assessment_id):
            return
        if args.auto_review:
            decision = {"decision": "accept", "reviewer": "auto (cli)", "comments": "auto-accepted"}
        else:
            print("\n=== HUMAN REVIEW REQUIRED ===")
            print(f"AI recommends: {report.recommendation} ({report.risk_rating})")
            choice = input("[a]ccept / [o]verride / [r]eturn for rework: ").strip().lower()[:1] or "a"
            decision = {"decision": {"a": "accept", "o": "override", "r": "return_for_rework"}.get(choice, "accept"),
                        "reviewer": input("Reviewer name: ").strip() or "cli-reviewer",
                        "comments": input("Comments: ").strip()}
            if decision["decision"] == "override":
                decision["final_recommendation"] = input("Final recommendation (Approve/Conditional/Reject): ").strip()
        final = await svc.review(report.assessment_id, decision)
        print(f"\nFinal status: {final.status} | human decision: {final.human_review.status} -> "
              f"{final.human_review.final_recommendation}")
        print(f"Saved: .data/assessments/{final.assessment_id}.json")

    asyncio.run(run())


def main() -> None:
    from .observability import setup_logging

    setup_logging()
    p = argparse.ArgumentParser(prog="it-agents", description="NFS vendor risk assessment deep agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="(Re)build the RAG index")
    a = sub.add_parser("assess", help="Run a vendor assessment")
    a.add_argument("--vendor", default="Asteria AI Systems")
    a.add_argument("--use-case", default="GenAI platform for processing and summarising confidential internal "
                                         "documents (vendor contracts, audit reports) for NFS risk & compliance teams.")
    a.add_argument("--classification", default="Confidential",
                   choices=["Public", "Internal", "Confidential", "Highly Confidential"])
    a.add_argument("--spend", type=float, default=None)
    a.add_argument("--pii", action="store_true", help="Vendor will handle customer PII")
    a.add_argument("--simulate-failure", action="append",
                   help="Force a tool or specialist to fail (e.g. get_financial_risk_intel, ai_governance)")
    a.add_argument("--auto-review", action="store_true", help="Auto-accept at the human review gate")
    s = sub.add_parser("serve", help="Start the web application")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    e = sub.add_parser("eval", help="Run the evaluation suite")
    e.add_argument("--case", action="append", help="Run only these case ids")
    args = p.parse_args()

    if args.cmd == "ingest":
        from .rag import ingest

        print(json.dumps(ingest(), indent=2))
    elif args.cmd == "assess":
        _assess(args)
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("it_agents.api:app", host=args.host, port=args.port, reload=args.reload,
                    reload_dirs=[str(Path(__file__).parent)] if args.reload else None)
    elif args.cmd == "eval":
        from .evals import run_suite

        asyncio.run(run_suite(args.case))
