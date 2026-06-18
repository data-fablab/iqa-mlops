# --- base stage: shared deps (numpy, pydantic, pillow) + project source ---
FROM python:3.12-slim AS base

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
# Config-driven boundaries read these at runtime (gates, monitoring thresholds).
COPY configs ./configs
RUN pip install --no-cache-dir uv && uv sync --no-dev

# Console scripts (iqa-api, iqa-run-ingestion, ...) are on PATH; no uv at runtime.
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app/src

# --- serving: iqa-api (no torch) ---
FROM base AS serving
RUN uv sync --no-dev --extra serving

# --- ml: iqa-inference, iqa-trainer (torch + scikit-learn + mlflow) ---
FROM base AS ml
ARG IQA_TORCH_EXTRA=cpu
RUN uv sync --no-dev --extra serving --extra ml --extra "$IQA_TORCH_EXTRA"

# --- data: iqa-ingestion, iqa-replay, iqa-monitoring (no torch) ---
FROM base AS data
RUN uv sync --no-dev --extra data
# Light, git-tracked manifests/plans the data boundaries validate at runtime
# (dataset/ingestion/replay). Heavy raw images (data/raw, DVC-managed) are NOT
# baked in -- they belong in MinIO/DVC (materialisation deferred, issues 18/19/20).
COPY data/metadata ./data/metadata
COPY data/model_datasets ./data/model_datasets

# --- dvc-gate: iqa-check-dvc-reproducibility (dvc[s3] + git, no torch) ---
FROM base AS dvc-gate
# dvc shells out to git; install the CLI (not the repo history -- .git is never
# baked in, so the gate runs the DVC/MinIO checks only, --skip-regeneration).
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN uv sync --no-dev --extra dvc
# DVC wiring the gate reads at runtime: the remote config, the pipeline file and
# the light *.dvc pointer (content-addressed; the heavy blob stays in MinIO).
# Never copy .dvc/cache or .dvc/config.local into the image.
COPY .dvc/config ./.dvc/config
COPY dvc.yaml ./dvc.yaml
COPY data/raw/hss-iad.dvc ./data/raw/hss-iad.dvc
