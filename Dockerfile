FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir uv && uv sync --extra cpu --no-dev

ENV PYTHONPATH=/app/src
