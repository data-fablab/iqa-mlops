FROM python:3.12-slim

ARG IQA_UV_EXTRA=cpu

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir uv && uv sync --extra "$IQA_UV_EXTRA" --no-dev

ENV PYTHONPATH=/app/src
