FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /usr/local/bin/uv
# the re-ranker's LightGBM needs OpenMP (Debian's libgomp1, as in Dockerfile.rerank)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# dependencies first so code edits don't bust the layer cache
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group rerank --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --group rerank

ENV PATH="/app/.venv/bin:$PATH" DATA_DIR=/data RESULTS_DIR=/results
RUN useradd --system --uid 10001 app && chown -R app /app
USER app
EXPOSE 8000
# state lives in Neo4j (VirtualQuery), so workers are interchangeable
# uvicorn kills a worker that misses one 5 s health ping; on a host under memory pressure
# that happened mid-eval (troubleshooting #19), so the supervisor waits 30 s
CMD ["uvicorn", "stunes.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-worker-healthcheck", "30"]
