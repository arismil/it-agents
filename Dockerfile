FROM python:3.14-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir uv && uv pip install --system .
EXPOSE 8000
CMD ["uvicorn", "it_agents.api:app", "--host", "0.0.0.0", "--port", "8000"]
