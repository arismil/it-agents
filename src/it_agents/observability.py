"""Basic observability: per-run event trace + metrics (always on) and Langfuse (optional)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import requests
from langchain_core.callbacks import AsyncCallbackHandler

from .config import AUDIT_LOG, RUNS_DIR

log = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    for noisy in ("httpx", "openai", "chromadb", "mcp", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RunTracer(AsyncCallbackHandler):
    """Collects LLM/tool events, token usage and latencies for one assessment run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.started = time.time()
        self.events: list[dict[str, Any]] = []
        self.tool_outputs: dict[str, str] = {}
        self._starts: dict[UUID, tuple[float, str, str]] = {}
        self.tokens = {"prompt": 0, "completion": 0}
        self.llm_calls = 0
        self.tool_calls = 0
        self.tool_errors = 0

    def event(self, kind: str, **data) -> None:
        self.events.append({"t": round(time.time() - self.started, 3), "kind": kind, **data})

    @staticmethod
    def _agent(metadata: dict | None) -> str:
        return (metadata or {}).get("lc_agent_name") or "orchestrator"

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        self._starts[run_id] = (time.time(), "llm", self._agent(metadata))

    async def on_llm_end(self, response, *, run_id, **kw):
        start, _, agent = self._starts.pop(run_id, (time.time(), "llm", "?"))
        self.llm_calls += 1
        usage = {}
        for gen in response.generations:
            for g in gen:
                msg = getattr(g, "message", None)
                if msg is not None and getattr(msg, "usage_metadata", None):
                    usage = msg.usage_metadata
        self.tokens["prompt"] += usage.get("input_tokens", 0)
        self.tokens["completion"] += usage.get("output_tokens", 0)
        self.event("llm_call", agent=agent, latency_s=round(time.time() - start, 2),
                   input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))

    async def on_llm_error(self, error, *, run_id, **kw):
        self._starts.pop(run_id, None)
        self.event("llm_error", error=str(error)[:300])

    async def on_tool_start(self, serialized, input_str, *, run_id, metadata=None, **kw):
        name = (serialized or {}).get("name", "?")
        self._starts[run_id] = (time.time(), name, self._agent(metadata))
        self.event("tool_start", agent=self._agent(metadata), tool=name, input=str(input_str)[:300])

    async def on_tool_end(self, output, *, run_id, **kw):
        start, name, agent = self._starts.pop(run_id, (time.time(), "?", "?"))
        self.tool_calls += 1
        text = output.content if hasattr(output, "content") else output
        text = text if isinstance(text, str) else json.dumps(text, default=str)
        self.tool_outputs[name] = self.tool_outputs.get(name, "") + "\n" + text
        is_error = "Error executing tool" in text[:200] or "ERROR:" in text[:20]
        self.tool_errors += is_error
        self.event("tool_end", agent=agent, tool=name, latency_s=round(time.time() - start, 2),
                   error=is_error, output_preview=text[:200])

    async def on_tool_error(self, error, *, run_id, **kw):
        _, name, agent = self._starts.pop(run_id, (0, "?", "?"))
        self.tool_errors += 1
        self.event("tool_error", agent=agent, tool=name, error=str(error)[:300])

    def metrics(self) -> dict[str, Any]:
        return {
            "latency_s": round(time.time() - self.started, 1),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "prompt_tokens": self.tokens["prompt"],
            "completion_tokens": self.tokens["completion"],
        }

    def save(self, extra: dict | None = None) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / f"{self.run_id}.json").write_text(json.dumps(
            {"run_id": self.run_id, "metrics": self.metrics(), "events": self.events, **(extra or {})},
            indent=2, default=str))


def load_trace(run_id: str) -> dict | None:
    p = RUNS_DIR / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def audit(run_id: str, event: str, **data) -> None:
    """Append-only audit trail (POL-AI-2025-01 §6 immutable logging, PoC version)."""
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as fh:
        fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                             "event": event, **data}, default=str) + "\n")


_langfuse_ok: bool | None = None


def langfuse_handler():
    """Return a Langfuse LangChain callback if enabled and reachable, else None."""
    global _langfuse_ok
    if os.getenv("LANGFUSE_TRACING_ENABLED", "false").lower() != "true" or not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return None
    if _langfuse_ok is None:
        host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "")
        try:
            _langfuse_ok = requests.get(f"{host}/api/public/health", timeout=2).ok
        except requests.RequestException:
            _langfuse_ok = False
        if not _langfuse_ok:
            log.warning("Langfuse enabled but unreachable at %s - continuing with local tracing only", host)
    if not _langfuse_ok:
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()
