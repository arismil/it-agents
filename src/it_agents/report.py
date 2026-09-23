"""Render the executive report as Markdown."""

from __future__ import annotations

from .models import AssessmentReport

_BADGE = {"retrieved_evidence": "EVIDENCE", "ai_inference": "INFERENCE", "missing_evidence": "MISSING"}


def to_markdown(r: AssessmentReport) -> str:
    L = [f"# Vendor Risk Assessment - {r.request.vendor_name}",
         f"*Assessment `{r.assessment_id}` | status **{r.status}** | generated {r.generated_at[:19]}Z*", ""]
    if r.status == "blocked":
        L += ["## Request blocked", r.executive_summary, ""]
        L += [f"- [{e.stage}] {e.type}: {e.detail}" for e in r.guardrail_events]
        return "\n".join(L)

    hr = r.human_review
    L += ["## Executive decision",
          "| | |", "|---|---|",
          f"| **AI Recommendation** | **{r.recommendation}** |",
          f"| **Risk rating** | **{r.risk_rating}** (VRS {r.risk_score}/100) |",
          f"| Vendor tier | {r.vendor_tier} |",
          f"| Human review | {hr.status}" + (f" by {hr.reviewer} -> **{hr.final_recommendation}**" if hr.reviewer else "") + " |",
          "", r.executive_summary, "", f"**Rationale:** {r.rationale}", ""]
    if hr.reasons:
        L += ["**Human review required because:**"] + [f"- {x}" for x in hr.reasons] + [""]
    if r.policy_violations:
        L += ["## Policy violations / veto triggers"] + [f"- {v}" for v in r.policy_violations] + [""]

    L += ["## Domain findings"]
    for d in r.domain_findings:
        L += [f"### {d.domain.replace('_', ' ').title()} - domain risk {d.risk_score}/100", d.summary, ""]
        for f in d.findings:
            veto = " **VETO**" if f.veto_trigger else ""
            L.append(f"- **[{_BADGE[f.evidence_type]}] [{f.severity.upper()}] {f.title}**{veto} "
                     f"(`{f.requirement_id}`; {f.compliance_status}; {f.policy_reference or 'n/a'})  ")
            L.append(f"  {f.detail}")
            for c in f.citations:
                L.append(f"  - `{c.source_id}`: \"{c.quote}\"")
        L.append("")
    for d in r.incomplete_domains:
        L += [f"### {d.replace('_', ' ').title()} - INCOMPLETE", "Specialist agent result unavailable.", ""]

    if r.cross_domain_contradictions:
        L += ["## Contradictions"] + [f"- {c}" for c in r.cross_domain_contradictions] + [""]
    L += ["## Missing evidence"] + [f"- ({m.domain}) {m.item}" for m in r.missing_evidence] + [""]
    L += ["## Required remediation"] + [f"{i}. {x}" for i, x in enumerate(r.required_remediation, 1)] + [""]

    valid = sum(c.valid for c in r.citations)
    L += [f"## Citations ({valid}/{len(r.citations)} verified)",
          "| Source | Document | Page | Verified | Note |", "|---|---|---|---|---|"]
    L += [f"| `{c.source_id}` | {c.document or '-'} | {c.page or '-'} | {'yes' if c.valid else 'NO'} | {c.reason} |"
          for c in r.citations]
    if r.guardrail_events:
        L += ["", "## Guardrail events"] + [f"- [{e.stage}] **{e.type}** ({e.action}): {e.detail}"
                                           for e in r.guardrail_events]
    if r.plan_steps_completed:
        L += ["", "## Execution plan (deep agent)"] + [f"- {s}" for s in r.plan_steps_completed]
    L += ["", "## Run metrics", "```", *[f"{k}: {v}" for k, v in r.metrics.items()], "```"]
    return "\n".join(L)
