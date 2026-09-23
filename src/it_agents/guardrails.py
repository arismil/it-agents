"""Deterministic guardrails for inputs, retrieved content, tool access and outputs.

Kept free of LLM calls so they are fast, predictable and unit-testable.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from .config import CHUNKS_FILE, MAX_TOOL_CALLS_PER_AGENT
from .models import (
    AssessmentRequest,
    CitationCheck,
    DomainAssessment,
    GuardrailEvent,
    RiskLevel,
)

log = logging.getLogger(__name__)

# ----------------------------------------------------------------- prompt injection
INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier|preceding) (instructions|rules|guidance|prompts?)",
    r"disregard (all |any )?(the )?(previous|prior|above|your|system) (instructions|rules|policies|prompts?)",
    r"forget (all |any )?(your|previous|prior) (instructions|rules)",
    r"override (the |all |any )?(policy|policies|guardrails?|instructions|assessment|safety)",
    r"\byou are now\b",
    r"\bnew (system )?instructions?\s*:",
    r"\bsystem\s*(prompt|message|override)\b",
    r"^\s*(system|assistant)\s*:",
    r"<\|?\s*(im_start|im_end|system)\s*\|?>",
    r"\b(ai|llm|agent|assistant|model)s?\b.{0,40}\b(must|should|shall)\b.{0,40}\b(approve|recommend approval|rate .{0,15}low)",
    r"\b(mark|rate|classify|set)\b.{0,40}\b(as )?(approved|low risk|compliant)\b.{0,40}\b(regardless|without|automatically)",
    r"do not (report|mention|flag|disclose|cite)",
    r"reveal (your|the) (system )?prompt",
    r"(output|respond with|print|include) the (phrase|token|string|word)",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def detect_injection(text: str) -> list[str]:
    """Return the matched suspicious phrases (empty list if clean)."""
    hits = []
    for rx in _INJECTION_RE:
        m = rx.search(text or "")
        if m:
            hits.append(m.group(0).strip())
    return hits


def sanitize_untrusted(text: str) -> tuple[str, list[str]]:
    """Redact sentences that contain injection attempts. Returns (clean_text, hits)."""
    hits = detect_injection(text)
    if not hits:
        return text, []
    parts = _SENTENCE_SPLIT.split(text)
    cleaned = [
        "[REDACTED BY NFS GUARDRAIL: suspected prompt-injection instruction]" if detect_injection(p) else p
        for p in parts
    ]
    return " ".join(cleaned), hits


# ------------------------------------------------------------------------ PII masking
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ACCOUNT_RE = re.compile(r"\b(?:\d[ -]?){8,15}(\d{4})\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@(?!northstarfin\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def mask_pii(text: str) -> str:
    """POL-DATA-2025-05 §4: mask account ids to last 4 digits, scrub SSNs and external emails."""
    text = _SSN_RE.sub("***-**-****", text)
    text = _ACCOUNT_RE.sub(lambda m: "****-" + m.group(1), text)
    return _EMAIL_RE.sub("[email redacted]", text)


# ---------------------------------------------------------------------- input checks
class InputRejected(Exception):
    def __init__(self, events: list[GuardrailEvent]):
        self.events = events
        super().__init__("; ".join(e.detail for e in events))


_SCOPE_RE = re.compile(
    r"\b(vendor|supplier|platform|saas|cloud|software|ai|genai|llm|service|tool|model|processing|assessment|document)s?\b",
    re.IGNORECASE,
)


def validate_request(req: AssessmentRequest) -> list[GuardrailEvent]:
    """Raise InputRejected for unsafe/out-of-scope requests; return informational events otherwise."""
    blocking: list[GuardrailEvent] = []
    for field in ("vendor_name", "use_case", "requested_by"):
        hits = detect_injection(getattr(req, field))
        if hits:
            blocking.append(GuardrailEvent(
                stage="input", type="prompt_injection",
                detail=f"Injection pattern in '{field}': {hits[0]!r}", action="blocked"))
    if not _SCOPE_RE.search(req.use_case):
        blocking.append(GuardrailEvent(
            stage="input", type="out_of_scope",
            detail="Use case does not describe a technology vendor/service to assess", action="blocked"))
    if not re.fullmatch(r"[\w .,&'()\-/]+", req.vendor_name):
        blocking.append(GuardrailEvent(
            stage="input", type="invalid_vendor_name",
            detail="Vendor name contains unsupported characters", action="blocked"))
    if blocking:
        raise InputRejected(blocking)

    events = []
    if mask_pii(req.use_case) != req.use_case:
        events.append(GuardrailEvent(stage="input", type="pii_masked",
                                     detail="PII detected in use case and masked before LLM dispatch", action="masked"))
    return events


def vendor_tier(req: AssessmentRequest) -> str:
    """POL-PROC-2025-02 §2 tiering (deterministic)."""
    spend = req.annual_spend_usd or 0
    # Tier 1 = spend > $250k OR customer PII OR AI models touching sensitive data.
    sensitive = req.data_classification in ("Confidential", "Highly Confidential")
    if spend > 250_000 or req.handles_customer_pii or sensitive:
        return "Tier 1 (Critical)"
    if spend >= 50_000:
        return "Tier 2 (Major)"
    return "Tier 3 (Standard)"


# ------------------------------------------------------------------- tool access
BUILTIN_TOOLS = {"write_todos", "read_file", "write_file", "edit_file", "ls", "glob", "grep", "task"}


def make_tool_guard(agent_name: str, allowed_tools: set[str], events: list[GuardrailEvent],
                    fail_agents: set[str] | None = None):
    """Middleware enforcing a per-agent tool allowlist + call budget, and converting
    tool/sub-agent exceptions into error ToolMessages so the run degrades gracefully."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    fail_agents = fail_agents or set()

    class ToolGuard(AgentMiddleware):
        name = f"ToolGuard[{agent_name}]"

        def __init__(self):
            super().__init__()
            self.calls: Counter[str] = Counter()

        async def abefore_agent(self, state, runtime):
            self.calls.clear()  # budget is per invocation, so a retried specialist starts fresh
            return None

        def _deny(self, request, msg: str, kind: str):
            events.append(GuardrailEvent(stage="tool", type=kind, detail=f"{agent_name}: {msg}", action="denied"))
            return ToolMessage(content=f"GUARDRAIL: {msg}", tool_call_id=request.tool_call["id"],
                               name=request.tool_call["name"], status="error")

        async def awrap_tool_call(self, request, handler):
            name = request.tool_call["name"]
            if name not in allowed_tools and name not in BUILTIN_TOOLS:
                return self._deny(request, f"tool '{name}' is not permitted for this agent", "tool_not_allowed")
            self.calls[name] += 1
            if sum(self.calls.values()) > MAX_TOOL_CALLS_PER_AGENT:
                return self._deny(request, "tool-call budget exhausted; finish with the evidence you have", "tool_budget")
            if name == "task":
                sub = request.tool_call["args"].get("subagent_type", "")
                if sub in fail_agents:
                    events.append(GuardrailEvent(stage="tool", type="agent_failure",
                                                 detail=f"Specialist '{sub}' failed (simulated outage)", action="degraded"))
                    return ToolMessage(
                        content=f"ERROR: specialist agent '{sub}' is unavailable (simulated outage). "
                                "Record this domain as incomplete / missing evidence and continue.",
                        tool_call_id=request.tool_call["id"], name=name, status="error")
            try:
                try:
                    result = await handler(request)
                except Exception as exc:
                    if name != "task":
                        raise
                    log.warning("specialist task failed in %s (%s); retrying once", agent_name, exc)
                    events.append(GuardrailEvent(stage="tool", type="agent_retry",
                                                 detail=f"{agent_name}: task failed ({exc}); retried once"[:300],
                                                 action="retried"))
                    result = await handler(request)
                content = str(getattr(result, "content", ""))
                if content.startswith("Error executing tool") or "simulated outage" in content[:300]:
                    events.append(GuardrailEvent(stage="tool", type="tool_error",
                                                 detail=f"{agent_name}: {content[:300]}", action="degraded"))
                return result
            except Exception as exc:  # graceful degradation: never crash the run on a tool error
                log.warning("tool %s failed in %s: %s", name, agent_name, exc)
                events.append(GuardrailEvent(stage="tool", type="tool_error",
                                             detail=f"{agent_name}: {name} failed: {exc}"[:400], action="degraded"))
                return ToolMessage(
                    content=f"ERROR: tool '{name}' failed ({type(exc).__name__}: {exc}). "
                            "Treat the requested evidence as MISSING and continue.",
                    tool_call_id=request.tool_call["id"], name=name, status="error")

    return ToolGuard()


# ------------------------------------------------------------- output validation
@lru_cache
def load_chunk_index() -> dict[str, dict[str, Any]]:
    index = {}
    if CHUNKS_FILE.exists():
        for line in CHUNKS_FILE.read_text().splitlines():
            rec = json.loads(line)
            index[rec["chunk_id"]] = rec
    return index


def _norm(s: str) -> str:
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[^a-z0-9%$.]+", " ", s.lower()).strip()


def quote_supported(quote: str, text: str, threshold: float = 0.8) -> bool:
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    q_tokens = q.split()
    t_tokens = set(t.split())
    return len(q_tokens) >= 3 and sum(tok in t_tokens for tok in q_tokens) / len(q_tokens) >= threshold


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    out = set()
    for n in _NUM_RE.findall(text):
        n = n.replace(",", "")
        out.add(n[:-2] if n.endswith(".0") else n)
    return out


def tool_quote_supported(quote: str, output: str) -> bool:
    """Tool outputs are JSON, so accept paraphrases whose figures all appear in the output."""
    material = {n for n in _numbers(quote) if float(n) >= 100}  # money / counts, not "3-year"
    if material:
        return material <= _numbers(output)
    return quote_supported(quote, output, 0.5)


def validate_citations(
    assessments: list[DomainAssessment],
    vendor_aliases: set[str],
    tool_outputs: dict[str, str],
) -> tuple[list[CitationCheck], list[GuardrailEvent]]:
    """Verify each citation exists in the indexed corpus (or an MCP tool output from this run)
    and that the quote is actually present. Unsupported 'retrieved_evidence' claims are
    downgraded to 'ai_inference' in place."""
    index = load_chunk_index()
    checks: list[CitationCheck] = []
    events: list[GuardrailEvent] = []
    aliases = {a.lower() for a in vendor_aliases}

    for da in assessments:
        for f in da.findings:
            any_valid = False
            for c in f.citations:
                sid = c.source_id.strip()
                doc, page, valid, reason = None, None, False, ""
                tool = re.sub(r"^(tool:|functions\.|mcp:)", "", sid)
                if sid not in index and (tool in tool_outputs or sid.startswith("tool:")):
                    out = tool_outputs.get(tool)
                    if out is None:
                        reason = "tool was not called in this run"
                    elif tool_quote_supported(c.quote, out):
                        valid, reason = True, "matches tool output"
                    else:
                        reason = "quote not found in tool output"
                elif sid in index:
                    rec = index[sid]
                    doc, page = rec["title"], rec["page"]
                    rec_vendor = (rec.get("vendor") or "").lower()
                    if rec["doc_type"] == "vendor" and rec_vendor not in aliases:
                        reason = f"cites another vendor's document ({rec.get('vendor')})"
                    elif quote_supported(c.quote, rec["text"]):
                        valid, reason = True, "quote verified in source chunk"
                    elif alt := next((k for k, r in index.items() if r["doc_id"] == rec["doc_id"]
                                      and quote_supported(c.quote, r["text"])), None):
                        valid, reason = True, f"quote verified in adjacent chunk {alt}"
                    else:
                        reason = "quote not found in cited chunk"
                else:
                    reason = "unknown source id (possible fabricated citation)"
                any_valid |= valid
                checks.append(CitationCheck(domain=da.domain, finding=f.title, source_id=sid,
                                            quote=c.quote, valid=valid, reason=reason, document=doc, page=page))
            if f.evidence_type == "retrieved_evidence" and not any_valid:
                f.evidence_type = "ai_inference"
                if f.veto_trigger:
                    f.veto_trigger = False
                    f.detail += " [Guardrail: veto claim not backed by a verified citation; requires human verification.]"
                events.append(GuardrailEvent(
                    stage="output", type="unsupported_claim",
                    detail=f"{da.domain}: '{f.title}' had no verifiable citation; relabelled as ai_inference",
                    action="relabelled"))
    return checks, events


# ------------------------------------------------------------ policy enforcement
def rating_for(score: int) -> RiskLevel:
    """POL-VRM-2025-03 §2 composite Vendor Risk Score thresholds."""
    if score <= 25:
        return "Low"
    if score <= 50:
        return "Moderate"
    if score <= 75:
        return "High"
    return "Critical"


def composite_score(assessments: list[DomainAssessment], incomplete: list[str]) -> int:
    scores = [min(max(a.risk_score, 0), 100) for a in assessments] + [70] * len(incomplete)  # unknown = high
    if not scores:
        return 100
    return round(0.5 * max(scores) + 0.5 * sum(scores) / len(scores))


def enforce_policy(
    llm_recommendation: str,
    assessments: list[DomainAssessment],
    incomplete: list[str],
    missing_count: int,
) -> tuple[str, RiskLevel, int, list[str], list[GuardrailEvent]]:
    """Deterministically reconcile the LLM's recommendation with NFS policy rules."""
    events: list[GuardrailEvent] = []
    score = composite_score(assessments, incomplete)
    # A veto needs all three: flagged by the specialist, backed by verified evidence, and actually non-compliant.
    vetoes = [f"{a.domain}: {f.title} ({f.policy_reference})" for a in assessments for f in a.findings
              if f.veto_trigger and f.evidence_type == "retrieved_evidence" and f.compliance_status == "non_compliant"]
    violations = vetoes + [f"{a.domain}: {f.title} ({f.policy_reference})" for a in assessments for f in a.findings
                           if f.compliance_status == "non_compliant" and not f.veto_trigger]
    if vetoes:
        score = max(score, 76)
    rating = rating_for(score)

    rec = llm_recommendation
    if vetoes or rating == "Critical":
        rec = "Reject"
        reason = "evidence-backed veto trigger(s): " + "; ".join(vetoes) if vetoes else f"composite score {score} is Critical"
    elif rec == "Approve" and (rating in ("High", "Moderate") or incomplete or missing_count):
        rec = "Conditional"
        reason = "Approve not permitted with elevated risk, incomplete domains or missing evidence"
    else:
        reason = ""
    if rec != llm_recommendation:
        events.append(GuardrailEvent(stage="policy", type="recommendation_override",
                                     detail=f"LLM proposed {llm_recommendation}; policy requires {rec}: {reason}",
                                     action="overridden"))
    return rec, rating, score, violations, events


def scan_output(text: str, canaries: list[str] | None = None) -> list[GuardrailEvent]:
    events = []
    hits = detect_injection(text)
    if hits:
        events.append(GuardrailEvent(stage="output", type="injection_echo",
                                     detail=f"Report text contains instruction-like content: {hits[0]!r}", action="flagged"))
    for c in canaries or []:
        if c.lower() in text.lower():
            events.append(GuardrailEvent(stage="output", type="canary_leak", detail=f"Canary {c} found in output",
                                         action="flagged"))
    return events
