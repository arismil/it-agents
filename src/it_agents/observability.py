from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


@contextmanager
def resolution_trace(trace_id: str, request: Any) -> Iterator[None]:
    """Create a Langfuse span when credentials are configured, otherwise no-op."""
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        with nullcontext():
            yield
        return

    from langfuse import get_client

    client = get_client()
    with client.start_as_current_observation(
        name="incident-resolution",
        as_type="chain",
        input=request.model_dump(),
        metadata={"trace_id": trace_id},
    ):
        yield
    client.flush()
