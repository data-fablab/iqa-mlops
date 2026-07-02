# Industrial Quality Assistant (IQA) — MLOps

![CI](https://github.com/data-fablab/iqa-mlops/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.12-blue)

Industrial Quality Assistant (IQA) is a completed MLOps MVP for visual quality
control on `Casting` parts. It packages a FastAPI application (`iqa-api`), a
separate inference service (`iqa-inference`), Streamlit review views for the Marc
and Sophie personas, replay runs, Airflow orchestration, DVC/MinIO data
reproducibility and lineage, MLflow tracking and registry, opt-in PostgreSQL
metadata persistence, Prometheus/Alertmanager/Grafana observability, and a
Kong/Nginx edge.

The MVP replays a historical Casting dataset through the same contracts a future
factory flow would use. In production, camera/MES adapters would emit
`production_ingest` events; in the school MVP, replay jobs emit `historical_replay`
events while preserving the same `piece_event` traceability chain.

## Project status

All three project phases are delivered and validated:

- **Phase 1 — Foundations.** Docker Compose stack, DVC + historical Casting data
  on MinIO, ROI bootstrap, Phase 1 API, feedback MVP, security and governance.
- **Phase 2 — Realistic loop.** Predictions/feedback/lots/incidents wired to
  PostgreSQL, candidate datasets from replay + oracle feedback, Feature-AE
  train/evaluate/log in MLflow, promotion and rollback gates via MLflow Registry,
  Airflow lifecycle and monitoring.
- **Phase 3 — Hardening.** Automated drift/incident scenarios, Grafana and
  Streamlit stabilisation, MinIO retention, validation reports, security controls,
  hardened team access and recovery procedures, Kong API gateway.
- **Phase 4 — Monitoring & lineage.** Prometheus scraping every service,
  Alertmanager drift/incident rules, four provisioned Grafana dashboards
  (overview, lifecycle, executive, drift), and an end-to-end data lineage from
  `sha256` to `feedback` — see [Observability & monitoring](#observability--monitoring)
  and [Data & lineage](#data--lineage) below.

### Out of scope (by design)

This is an academic MVP, not a production deployment. The following are
intentionally excluded — see [docs/roadmap-iqa.md](docs/roadmap-iqa.md):

- Kubernetes (the stack runs on Docker Compose).
- Full OAuth/RBAC (edge auth is handled at the Kong/Nginx layer).
- Automated retraining of the ROI segmenter.
- One `pyproject.toml` per service (a single root project is used — see
  [ADR 0007](docs/adr/0007-architecture-services-avec-pyproject-racine.md)).

Object storage is local MinIO; no paid cloud storage is required.

## Architecture

```text
                         Kong / Nginx (edge)
                                 |
                +----------------+----------------+
                |                                 |
            iqa-api  <---------- REST ----------> iqa-inference
                |                                 |
     +----------+----------+          +-----------+-----------+
     |          |          |          |                       |
 PostgreSQL   MinIO      MLflow     MinIO                   MLflow
 (metadata) (images)   (registry) (checkpoints)           (artifacts)

 Airflow orchestrates:  ingestion · replay · monitoring · lifecycle · dvc-repro
 Prometheus + Alertmanager + Grafana observe every service
```

Full topology, data contracts, and component responsibilities:
[docs/architecture-iqa.md](docs/architecture-iqa.md).

The Phase 2/3 traceability chain — `piece_event` is the atomic unit for split,
replay, validation, feedback, and training eligibility:

```text
sha256 -> piece_event -> scenario -> lot -> dataset_version -> model_version -> prediction -> feedback
```

## MLOps practices

| Practice | How it is implemented | Evidence |
| --- | --- | --- |
| Data reproducibility & lineage | DVC with a MinIO remote (`iqa-minio` → `s3://iqa-dvc`), enforced by the `iqa_dvc_reproducibility` DAG as an explicit gate | [dvc-versioning.md](docs/dvc-versioning.md), [lineage-evidence.md](docs/lineage-evidence.md) |
| Model registry as source of truth | MLflow Registry decides the active model; promotion and rollback go through it, MinIO only stores artifacts | [mlflow-registry.md](docs/mlflow-registry.md), [ADR 0006](docs/adr/0006-mlflow-registry-source-verite.md) |
| Promotion gates & rollback | Config-driven gates (`iqa-run-gates`, `iqa-run-promotion`); model rollback via Registry, app rollback by immutable image tag | [gates.md](docs/gates.md), [rollback.md](docs/rollback.md), [rollback-server.md](docs/rollback-server.md) |
| CI/CD | 4-job GitHub Actions: lint+test, API/DAG contracts (zero-broken-DAG check), docker build + compose validate, opt-in image publish with immutable tags (git SHA + `v*`, never `latest`) | [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| Automated testing | 764 tests across 107 files, including API/data/contract tests and Airflow DAG unit checks | [tests/](tests/) |
| Orchestration | 9 Airflow DAGs (ingestion, replay, monitoring, lifecycle, drift, DVC reproducibility) run as containerised tasks | [ADR 0002](docs/adr/0002-airflow-comme-orchestrateur.md), [ADR 0008](docs/adr/0008-taches-airflow-comme-conteneurs.md) |
| Observability & monitoring | Prometheus scraping every service, Alertmanager drift rules, 4 provisioned Grafana dashboards | [Observability & monitoring](#observability--monitoring), [deploy/prometheus](deploy/prometheus), [deploy/grafana](deploy/grafana) |
| Reproducible environments | `uv` lockfile with role/CUDA extras, multi-stage Dockerfile, per-role images | [pyproject.toml](pyproject.toml), [Dockerfile](Dockerfile) |
| API gateway & edge | Kong gateway (Phase 3) in front of the services | [api-gateway.md](docs/api-gateway.md), [ADR 0009](docs/adr/0009-kong-api-gateway-phase3.md) |
| Security & governance | Audit trail, AI security governance, feedback eligibility rules, documented decisions (ADR 0001–0009) | [ai_security_governance.md](docs/ai_security_governance.md), [audit_trail.md](docs/audit_trail.md), [docs/adr/](docs/adr/) |

## Observability & monitoring

Every service exposes a Prometheus text endpoint and is scraped centrally;
Alertmanager fires on drift/incident conditions and Grafana renders the story.

**Prometheus scrape targets** ([deploy/prometheus/prometheus.yml](deploy/prometheus/prometheus.yml)):

| Target | Endpoint | Interval |
| --- | --- | --- |
| `iqa-api` | `:8000/metrics` | 5s |
| `iqa-inference` | `:8100/metrics` | 15s |
| Airflow | `statsd-exporter:9102/metrics` (StatsD → Prometheus sidecar) | 15s |
| MinIO | `:9000/minio/v2/metrics/cluster` | 15s |

**Metric families** exposed by the application (`iqa_*`):

- Service health — `iqa_api_up`, `iqa_inference_up`, `iqa_active_model_info`,
  `iqa_inference_gpu_lock_held`.
- Drift — `iqa_drift_score`, `iqa_drift_status`, `iqa_drift_degradation_score`,
  `iqa_drift_oracle_fn_rate`, `iqa_drift_red_rate`, `iqa_drift_trigger_lifecycle`.
- Lifecycle — `iqa_lifecycle_cycle_current`, `iqa_lifecycle_epoch_image_ap`,
  `iqa_lifecycle_epoch_pixel_aupimo`, `iqa_lifecycle_active_model_info`.
- Quality & feedback — `iqa_feedback_conflict_total`, `iqa_invalid_feedback_total`,
  `iqa_divergence_filtered_total`.
- Security — `iqa_ai_security_incident_total`.

**Alerting** — Alertmanager ([deploy/alertmanager/alertmanager.yml](deploy/alertmanager/alertmanager.yml))
with recorded/alerting rules for the drift proxy and the `piece_a → P4` drift
scenario ([deploy/prometheus/rules/](deploy/prometheus/rules/), rules
`IqaDriftProxy`, `IqaPieceAP…`).

**Grafana dashboards** — auto-provisioned datasource and dashboards
([deploy/grafana/provisioning/](deploy/grafana/provisioning/)):

| Dashboard | Purpose |
| --- | --- |
| `iqa-overview` | Service health, prediction latency, V/O/R decision mix, incidents |
| `iqa-lifecycle` | The industrial MLOps "red thread": scenario cycles, retrain epochs, promotions |
| `iqa-executive-mlops` | High-level MLOps posture for a non-technical audience |
| `iqa-drift-p4` | Drift regime for the `piece_a → P4` scenario and lifecycle triggers |

Bring the observability stack up and operate it via
[docs/exploitation-runbook.md](docs/exploitation-runbook.md); drift regimes are
documented in [docs/drift-regimes.md](docs/drift-regimes.md).

## Quick start

Local validation:

```bash
uv sync --extra cpu
uv run --extra cpu pytest -q
uv run --extra cpu ruff check src scripts tests
```

Local API:

```bash
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

### Common commands

```bash
uv sync --extra cpu                                   # install (CPU torch)
uv run --extra cpu pytest -q                          # run the test suite
uv run --extra cpu ruff check src scripts tests       # lint
uv run --extra cpu iqa-api                             # run the API locally
docker compose --env-file ../.env up -d               # run the stack (from deploy/)
uv run --extra cpu iqa-demo-phase2                     # end-to-end demo
```

The full CLI surface (~50 `iqa-*` entry points) is declared in
[pyproject.toml](pyproject.toml) under `[project.scripts]`; every command accepts
`--help`. Operational procedures live in
[docs/exploitation-runbook.md](docs/exploitation-runbook.md) and
[docs/replay-runbook.md](docs/replay-runbook.md).

## API

Public endpoints:

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

Internal-only routes (`/internal/drift/events`, `/internal/lifecycle/events`) are
called by orchestration and are not part of the public surface. Critical
scenario-scoped routes keep `scenario_id` mandatory to preserve isolation between
`production_replay_natural`, `drift_domain_extension`, and future production
scenarios. Full contracts: [docs/api_contracts.md](docs/api_contracts.md).

## Data & lineage

Every artifact is traceable end to end. The chain is
`sha256 -> piece_event -> scenario -> lot -> dataset_version -> model_version -> prediction -> feedback`,
and each stage has a producer and a store:

| Stage | Meaning | Produced by | Recorded in |
| --- | --- | --- | --- |
| `sha256` | Content hash of the raw image | Ingestion/replay | Manifest + PostgreSQL |
| `piece_event` | Atomic traceable piece | Ingestion contract | PostgreSQL (`piece_events`) |
| `scenario` | Replay/production context | Replay driver | `piece_event.scenario_id` |
| `lot` | Batch grouping for review | Aggregation | PostgreSQL (`lot_events`) |
| `dataset_version` | Frozen candidate dataset | Dataset build (DVC) | DVC + MinIO |
| `model_version` | Trained & promoted model | Training + MLflow Registry | MLflow (`model_version_events`) |
| `prediction` | Inference result on a piece | `iqa-inference` | PostgreSQL (`predictions`) |
| `feedback` | Oracle/review verdict | `oracle_gt` / review | PostgreSQL (`feedback_events`) |

Reproduce and audit the full chain with
[docs/lineage-evidence.md](docs/lineage-evidence.md) and the
[docs/phase3-final-lineage-runbook.md](docs/phase3-final-lineage-runbook.md);
`iqa-lineage-summary` renders it on demand.

`piece_event` is the atomic split, replay, validation, feedback, and training
eligibility unit. Supported replay scenarios:

- `production_replay_natural`
- `drift_domain_extension`

`bootstrap`, `calibration_good_reference_v001`, replay manifests, and
`validation_set_replay_representative_v001` remain disjoint. `oracle_gt` is the
sovereign feedback source for training eligibility; Sophie remains a
display/review persona in this phase. Feature-AE MVP training uses one stable good
anchor, `feature_ae_good_mvp_v001`, disjoint from validation, calibration and
replay.

Model lifecycle decisions are triggered by data events (e.g. 50 new
oracle-validated conforming pieces, or confirmed drift). CI validates contracts
and builds images, but it does not trigger model training. See
[docs/model-lifecycle.md](docs/model-lifecycle.md) and
[docs/feedback_rules.md](docs/feedback_rules.md).

## Storage & repository state

The source dataset and ingested runtime images are intentionally separate:

- `s3://iqa-source-datasets/hss-iad-casting-raw-v1` — the immutable source
  dataset used by replay and inventory jobs.
- `s3://iqa-ingested-images/...` — images after they pass the ingestion contract
  and have an associated `piece_event`.

PostgreSQL stores metadata facts, statuses, timestamps, versions, URIs, and JSONB
payloads — never binary artifacts. Runtime PostgreSQL write-through is explicit
and opt-in via `IQA_METADATA_BACKEND=postgres`. Model checkpoints are restored
from MinIO manifests into `.cache/iqa/models/`; the `models/` tree stores
manifests only.

**Tracked by Git:** source code, tests, docs, configs, lightweight CSV/model
manifests, DVC metadata and reproducibility contracts. No PyTorch checkpoints, no
generated local metadata databases.

**Stored outside Git:** checkpoints and model artifacts in MinIO
(`s3://iqa-models`, `s3://mlflow-artifacts`), replayed/production raw images
(`s3://iqa-ingested-images`), heavy data via DVC/MinIO, and runtime metadata in
PostgreSQL when the opt-in backend is enabled.

## Documentation

Start here: [docs/index.md](docs/index.md).

Core:

- [docs/architecture-iqa.md](docs/architecture-iqa.md) — architecture
- [docs/prd-iqa-mvp.md](docs/prd-iqa-mvp.md) — product & MVP scope
- [docs/api_contracts.md](docs/api_contracts.md) — API contracts
- [docs/data-contracts.md](docs/data-contracts.md) — data contracts
- [docs/dvc-versioning.md](docs/dvc-versioning.md) — DVC versioning
- [docs/roadmap-iqa.md](docs/roadmap-iqa.md) — roadmap & scope

Operations (Phase 3):

- [docs/deploy_runbook.md](docs/deploy_runbook.md) — full deployment
- [docs/exploitation-runbook.md](docs/exploitation-runbook.md) — day-to-day ops
- [docs/replay-runbook.md](docs/replay-runbook.md) — replay runs (API + Airflow)
- [docs/rollback.md](docs/rollback.md) / [docs/rollback-server.md](docs/rollback-server.md) — model & app rollback
- [docs/retention_storage.md](docs/retention_storage.md) — object storage & retention
- [docs/api-gateway.md](docs/api-gateway.md) — Kong API gateway

Governance & models:

- [docs/ai_security_governance.md](docs/ai_security_governance.md) — AI security & governance
- [docs/audit_trail.md](docs/audit_trail.md) — audit & traceability
- [docs/adr/](docs/adr/) — architecture decision records (0001–0009)
- [docs/modele-feature-ae-iqa.md](docs/modele-feature-ae-iqa.md) — Feature-AE model
- [docs/modele-segmentation-roi-iqa.md](docs/modele-segmentation-roi-iqa.md) — ROI segmenter

## License

MIT — see [LICENSE](LICENSE).
