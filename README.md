# Industrial Quality Assistant MLOps

Industrial Quality Assistant (IQA) is a Phase 2 MLOps MVP for visual quality
control on `Casting` parts. It combines a FastAPI application (`iqa-api`), a
separate inference service (`iqa-inference`), Streamlit review views for Marc
and Sophie, replay runs, Airflow orchestration, DVC/MinIO data reproducibility,
MLflow tracking and registry, optional PostgreSQL metadata persistence,
Prometheus/Grafana observability, and an Nginx reverse proxy.

The MVP replays a historical Casting dataset through the same contracts expected
from a future factory flow. In production, camera/MES adapters will emit
`production_ingest` events. In the school MVP, replay jobs emit
`historical_replay` events while preserving the same `piece_event` traceability
chain.

## Historical Dataset Vs Ingested Data

The source dataset and ingested runtime images are intentionally separate:

- `s3://iqa-source-datasets/hss-iad-casting-raw-v1` stores the immutable source
  dataset used by replay and inventory jobs.
- `s3://iqa-ingested-images/...` stores images after they pass through the
  ingestion contract and have an associated `piece_event`.

For the student MVP, these buckets are hosted by local MinIO. No paid cloud
storage is required.

## Quick Start

Local validation:

```powershell
uv sync --extra cpu
uv run --extra cpu pytest -q
uv run --extra cpu ruff check src scripts tests
```

Local API:

```powershell
uv run --extra cpu iqa-api
```

Docker Compose stack:

```bash
cd deploy
docker compose --env-file ../.env up -d postgres minio minio-init
docker compose --env-file ../.env up -d iqa-inference iqa-api
```

For the full server sequence, GPU overlay, Airflow, observability, smoke tests,
and rollback notes, see [docs/deploy_runbook.md](docs/deploy_runbook.md).

## Phase 2 API

Available endpoints include:

- `GET /health`
- `GET /model/version`
- `POST /predict`
- `POST /piece-events/{event_id}/predict`
- `GET /replay-scenarios`
- `POST /replay-runs`
- `GET /replay-runs/{replay_run_id}/next`
- `POST /replay-runs/{replay_run_id}/reset`
- `POST /feedback`
- `GET /predictions`
- `GET /lots/summary`
- `GET /incidents`
- `GET /metrics`
- `POST /admin/reload-model`

Critical scenario-scoped routes keep `scenario_id` mandatory. This preserves
isolation between `production_replay_natural`, `drift_domain_extension`, and
future production scenarios.

## Public Commands

Data, manifests, and replay:

```powershell
uv run --extra cpu iqa-build-inventory --help
uv run --extra cpu iqa-finalize-data-phase1 --help
uv run --extra cpu iqa-build-flux-plan --help
uv run --extra cpu iqa-build-feature-ae-datasets --help
uv run --extra cpu iqa-simulate-lifecycle --help
uv run --extra cpu iqa-prepare-sim-env --help
uv run --extra cpu iqa-validate-mvp --help
uv run --extra cpu iqa-validate-ml-source --help
```

DVC/MinIO reproducibility:

```powershell
uv run --extra cpu --extra data iqa-check-dvc-reproducibility --help
```

PostgreSQL metadata:

```powershell
uv run --extra cpu iqa-init-metadata-db --help
```

Runtime services:

```powershell
uv run --extra cpu iqa-api --help
uv run --extra cpu iqa-inference --help
uv run --extra cpu iqa-predict-image --help
uv run --extra cpu iqa-predict-roi --help
uv run --extra cpu iqa-generate-bootstrap-roi --help
```

Airflow boundary scripts:

```powershell
uv run --extra cpu iqa-run-ingestion --help
uv run --extra cpu iqa-run-replay --help
uv run --extra cpu iqa-run-monitoring --help
uv run --extra cpu iqa-run-lifecycle --help
```

Phase 2 demo:

```powershell
uv run --extra cpu iqa-demo-phase2 --help
```

## Architecture And Storage

Git tracks source code, tests, documentation, configuration, lightweight CSV
manifests, model manifests, and DVC metadata.

DVC and MinIO handle heavy data and versioned data artifacts. The default DVC
remote is `iqa-minio` targeting `s3://iqa-dvc`, and the Airflow DAG
`iqa_dvc_reproducibility` exposes DVC as an explicit reproducibility and data
lineage gate.

MLflow is the tracking and registry source of truth for model promotion and
rollback. MinIO stores MLflow artifacts and model files; it does not decide
which model is active.

PostgreSQL stores metadata facts, statuses, timestamps, versions, URIs, and JSONB
payloads. It never stores images, checkpoints, masks, heatmaps, or other binary
artifacts. Runtime PostgreSQL write-through remains explicit and opt-in through
`IQA_METADATA_BACKEND=postgres`.

## Data, Replay And Lifecycle

`piece_event` is the atomic split, replay, validation, feedback, and training
eligibility unit. The Phase 2 traceability chain is:

```text
sha256 -> piece_event -> scenario -> lot -> dataset_version -> model_version -> prediction -> feedback
```

The supported replay scenarios are:

- `production_replay_natural`
- `drift_domain_extension`

`bootstrap`, `calibration_set_v001`, replay manifests, and
`validation_set_v001` remain disjoint. `oracle_gt` is the sovereign feedback
source for training eligibility; Sophie remains a display/review persona in this
phase.

Feature-AE candidate datasets are materialized from oracle-validated conforming
pieces:

- `feature_ae_good_v002` from natural replay conforming pieces.
- `feature_ae_good_v003` from drift/domain-extension conforming pieces.

Model lifecycle decisions are triggered by data events, such as 50 new
oracle-validated conforming pieces or confirmed drift. CI validates contracts and
builds images, but it does not trigger model training.

## Repository State

Tracked by Git:

- source code, tests, docs, configs, lightweight manifests, model manifests;
- DVC metadata and reproducibility contracts;
- no PyTorch checkpoints or generated local metadata databases.

Stored outside Git:

- checkpoints and model artifacts in MinIO, especially `s3://iqa-models` and
  `s3://mlflow-artifacts`;
- production or replayed raw images under `s3://iqa-ingested-images`;
- heavy data and versioned data artifacts through DVC/MinIO;
- runtime metadata in PostgreSQL when the opt-in backend is enabled.

## Main Documentation

- [docs/index.md](docs/index.md)
- [docs/architecture-iqa.md](docs/architecture-iqa.md)
- [docs/api_contracts.md](docs/api_contracts.md)
- [docs/data-contracts.md](docs/data-contracts.md)
- [docs/dvc-versioning.md](docs/dvc-versioning.md)
- [docs/replay-runbook.md](docs/replay-runbook.md)
- [docs/deploy_runbook.md](docs/deploy_runbook.md)
- [docs/retention_storage.md](docs/retention_storage.md)
- [docs/ai_security_governance.md](docs/ai_security_governance.md)

Model-specific contracts:

- [docs/modele-feature-ae-iqa.md](docs/modele-feature-ae-iqa.md)
- [docs/modele-segmentation-roi-iqa.md](docs/modele-segmentation-roi-iqa.md)
