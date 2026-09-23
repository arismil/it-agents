FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY evals ./evals
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" NFS_DATA_DIR=/app/.data
EXPOSE 8000
# The RAG index is built on first start if missing (persisted in the /app/.data volume).
CMD ["it-agents", "serve", "--host", "0.0.0.0", "--port", "8000"]
