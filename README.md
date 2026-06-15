# Industrial Quality Assistant MLOps

Industrial Quality Assistant (IQA) is an MLOps MVP for visual quality control on
`Casting` parts. The repository currently provides the Phase 1 foundation:
reproducible Python environment, model packaging code, data/replay manifests,
simulation scripts, validation artifacts, and a minimal FastAPI skeleton.

The project metadata store target is PostgreSQL. Local generated databases and
service artifacts stay outside Git.

## Historical Dataset vs Ingested Data

The Casting dataset is treated as historical plant data, not as the production
store itself. For the school MVP, replay jobs emit `historical_replay` piece
events through the same ingestion contract expected from a future factory flow.
In production, camera/MES adapters will emit `production_ingest` events. The
runtime target is: raw images go to MinIO, especially
`s3://iqa-ingested-images`, while PostgreSQL stores piece events, timestamps,
model links, predictions, feedback, and artifact URIs.

The source dataset and the ingested images are intentionally separate:

- `s3://iqa-source-datasets/hss-iad-casting-raw-v1` stores the immutable source
  dataset used by replay and inventory jobs.
- `s3://iqa-ingested-images/...` stores images after they have passed through the
  ingestion contract and have an associated `piece_event`.

For the student MVP, both buckets are hosted by local MinIO on the workstation.
No paid cloud storage is required.

## Quick Start

```powershell
uv sync --extra cpu
uv run --extra cpu pytest -q
uv run --extra cpu ruff check src scripts tests
```

## Public Commands

```powershell
uv run --extra cpu iqa-build-inventory --help
uv run --extra cpu iqa-finalize-data-phase1 --help
uv run --extra cpu iqa-build-flux-plan --help
uv run --extra cpu iqa-simulate-lifecycle --help
uv run --extra cpu iqa-prepare-sim-env --help
uv run --extra cpu iqa-validate-mvp --help
uv run --extra cpu iqa-validate-ml-source --help
uv run --extra cpu iqa-train-feature-ae --help
uv run --extra cpu iqa-evaluate-feature-ae --help
uv run --extra cpu iqa-predict-image --help
uv run --extra cpu iqa-predict-roi --help
uv run --extra cpu iqa-generate-bootstrap-roi --help
uv run --extra cpu iqa-api --help
uv run --extra cpu iqa-inference --help
uv run --extra cpu iqa-run-ingestion --help
uv run --extra cpu iqa-run-replay --help
uv run --extra cpu iqa-run-monitoring --help
uv run --extra cpu iqa-run-lifecycle --help
```

See [docs/reproductibilite-ml-iqa.md](docs/reproductibilite-ml-iqa.md) for the
source-to-prediction path: dataset source, ingestion/manifests, Feature-AE train,
checkpoint, and image prediction. The retained Feature-AE preprocessing uses
`tiled_context` with explicit `image_size=384` and `context_size=768`; the old
source name `tile_256_overlap` is intentionally not reused.

Model-specific contracts are documented in:
- [docs/modele-feature-ae-iqa.md](docs/modele-feature-ae-iqa.md)
- [docs/modele-segmentation-roi-iqa.md](docs/modele-segmentation-roi-iqa.md)

## API Skeleton

```powershell
uv run --extra cpu uvicorn iqa.api.main:app --host 0.0.0.0 --port 8000
```

Available now:
- `GET /health`
- `GET /model/version`
- `POST /predict` as an explicit Phase 1 placeholder

## Repository State

Tracked by Git:
- source code, tests, docs, configs, small data manifests, model manifests;
- no PyTorch checkpoints or generated local metadata databases.

Stored outside Git:
- checkpoints in MinIO under `s3://iqa-models`;
- production or replayed raw images in MinIO under `s3://iqa-ingested-images`;
- source datasets and heavy data through DVC/MinIO in later phases.

## Convergence Decisions

The repo keeps a root `pyproject.toml` for the initial deliverable while Docker
Compose separates the target services. The official serving boundary remains
`iqa-api` plus a separate `iqa-inference` service.

Data and model lifecycle rules:

- `piece_event` is the atomic split unit.
- `calibration_set_v001` is a good-only set, sealed away from bootstrap, replay,
  training, and `validation_set_v001`.
- Replay events carry `event_time`, `recorded_at`, and derived `is_simulated`.
- MLflow Registry is the target source of truth for promotion and rollback.
- Registered models are isolated by scenario:
  `feature_ae__production_replay_natural` and
  `feature_ae__drift_domain_extension`.
- Sophie is a MVP showcase interface; `oracle_gt` drives the automated workflow.
