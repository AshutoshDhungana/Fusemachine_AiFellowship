# ---- API image (FastAPI + RAG + ONNX router). No torch at runtime.
FROM python:3.11-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/cache/hf FASTEMBED_CACHE_PATH=/cache/fastembed
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY prompts ./prompts
COPY data/papers.yaml ./data/papers.yaml
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "assistant.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
