# Airflow Container Runtime Evidence

This document captures the Phase 3 evidence that Airflow orchestrates IQA
application workflows as containers, as required by ADR 0008. Docker Compose
orchestre les services longs (API, inference, MinIO, PostgreSQL, MLflow,
Grafana, Streamlit, gateway). Airflow orchestre les workflows metier
applicatifs. Airflow imports DAG files and the lightweight `iqa.dags` factory
only; the metier runtime runs inside task images.

Docker Compose orchestre les services longs ; Airflow orchestre les workflows
metier applicatifs.

## Runtime Boundary

- `iqa_ingestion`, `iqa_replay`, `iqa_monitoring`, `iqa_lifecycle`,
  `iqa_lifecycle_trigger` and `iqa_drift_piece_a_p4` use `make_container_task`.
- `make_container_task` currently builds DockerOperator tasks when
  `IQA_AIRFLOW_BACKEND=docker`.
- Task containers join `iqa_net` so they can resolve `postgres`, `minio`,
  `mlflow`, `iqa-api` and `iqa-inference`.
- The Feature-AE application lifecycle runs `iqa-run-replay-lifecycle-cycle`
  in the ML image. This is the pipeline applicatif Feature-AE de reference:
  replay, progressive training, fair active-vs-candidate comparison, MLflow
  evidence and test-stage promotion.
- The application lifecycle task uses Airflow pool `iqa_gpu` and the shared
  `iqa_gpu_lock` volume.
- `iqa_lifecycle` has `max_active_runs=1`, an explicit execution timeout and no
  retry loop for long GPU training runs.
- `iqa_dvc_reproducibility` remains a separate DVC gate. It does not run `dvc
  push` and does not trigger a model lifecycle.

## Security Boundary

The Docker backend is the validated Phase 3 backend. The scheduler mounts
`/var/run/docker.sock`, which is a strong host privilege and must stay limited to
the trusted MVP server. Kubernetes reste Phase 4; the Kubernetes backend remains
an escape hatch/stub, not a validated production path.

The task environment is allowlisted by the factory. Secrets and service URLs are
forwarded only when their names are explicitly accepted by
`DEFAULT_TASK_ENV_PASSTHROUGH` or `IQA_TASK_ENV_PASSTHROUGH`.

## Static Evidence

Run from the repository root:

```bash
uv run --extra cpu iqa-check-airflow-container-runtime --json
```

Expected result:

```json
{
  "backend": "docker",
  "dvc_gate": "iqa_dvc_reproducibility",
  "gpu_pool": "iqa_gpu",
  "lifecycle_command": "iqa-run-replay-lifecycle-cycle",
  "lifecycle_mode": "progressive-train",
  "network": "iqa_net",
  "promotion_policy": "candidate_must_improve_active_on_same_eval_set",
  "registry_stage": "test",
  "status": "validated"
}
```

## Server Evidence

Run from the server repository:

```bash
cd /opt/iqa/iqa-mlops
git fetch origin
git checkout feature/airflow-container-runtime-evidence
git pull --ff-only

docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml config -q

docker compose exec airflow-webserver airflow dags list
docker compose exec airflow-webserver airflow dags list-import-errors
docker compose exec airflow-webserver airflow pools list

docker compose exec airflow-webserver airflow dags unpause iqa_dvc_reproducibility
docker compose exec airflow-webserver airflow dags unpause iqa_lifecycle_trigger
docker compose exec airflow-webserver airflow dags unpause iqa_drift_piece_a_p4

docker compose exec airflow-webserver airflow dags trigger iqa_dvc_reproducibility \
  --conf '{"with_network": false,"skip_regeneration": true}'

docker compose exec airflow-webserver airflow dags trigger iqa_lifecycle_trigger \
  --conf '{"scenario_id":"production_replay_natural","conforming_validated_count":50,"drift_confirmed":false,"roi_fail_rate":0.0}'

docker compose exec airflow-webserver airflow dags trigger iqa_drift_piece_a_p4

docker compose exec airflow-webserver airflow dags unpause iqa_lifecycle
docker compose exec airflow-webserver airflow dags trigger iqa_lifecycle \
  --conf '{"mode":"progressive-train","max_events":260,"lifecycle_interval":50,"max_cycles":3,"epochs":10,"target_stage":"test","promotion_min_delta":0.0}'
```

Expected evidence:

- the DAG list includes `iqa_ingestion`, `iqa_replay`, `iqa_monitoring`,
  `iqa_lifecycle`, `iqa_lifecycle_trigger`, `iqa_drift_piece_a_p4` and
  `iqa_dvc_reproducibility`;
- `airflow dags list-import-errors` is empty;
- `airflow pools list` includes `iqa_gpu`;
- the DAGs used for the proof are unpaused before triggering;
- `iqa_dvc_reproducibility` can be triggered explicitly;
- `iqa_lifecycle_trigger` forwards a data event to `iqa_lifecycle`;
- `iqa_drift_piece_a_p4` observes the Piece B -> Piece A/P4 replay before
  triggering one correction lifecycle;
- `iqa_lifecycle` runs the application lifecycle task `run_application_lifecycle`;
- there is pas de training via CI.

## What This Does Not Change

This evidence does not promote models, does not trigger lifecycle from CI, does
not change the runtime API, and does not move binary artifacts into Git. Model
truth remains in MLflow Registry, checkpoints remain in MinIO, and DVC remains a
reproducibility gate.
