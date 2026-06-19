# Phase 3 Final Lineage Runbook

This runbook is the short soutenance and handover path for IQA Phase 3. It
connects the deployed stack, Airflow orchestration, DVC lineage, MinIO
artifacts, MLflow Registry evidence and the business/security reading of the
demo.

## 1. Server Prerequisites

Start from the server repository and a published Docker Hub image tag:

```bash
cd /opt/iqa/iqa-mlops
git checkout main
git pull --ff-only

export IQA_IMAGE_REGISTRY=adrien1101
export IQA_IMAGE_TAG=sha-$(git rev-parse HEAD)
export IQA_DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
```

`IQA_IMAGE_REGISTRY` is replaceable by the production organization namespace.
The Phase 3 demo namespace is `adrien1101`. `IQA_DOCKER_GID` must match the host
Docker socket group so Airflow `DockerOperator` tasks can launch containers.
The deployment tag must follow the immutable CI SHA convention:
`IQA_IMAGE_TAG=sha-<commit>`.

Authenticate the server to Docker Hub:

```bash
docker login
```

## 2. Deploy From Published Images

The deployment proof is `pull -> up -d -> smoke`; the server must not rebuild IQA
application images for this proof.

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml pull
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml up -d
bash deploy/smoke-test.sh
```

Expected result:

```text
RESULTAT : tous les smoke tests sont verts.
```

This proves Docker Hub images, API, inference, MinIO, MLflow, Prometheus,
Grafana, Airflow and gateway routes are operational.

## 3. Airflow Container Runtime Evidence

Airflow must orchestrate containers, not import the business runtime in the
scheduler:

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml exec airflow-webserver airflow dags list
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml exec airflow-webserver airflow dags list-import-errors
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml exec airflow-webserver airflow pools list
```

Expected proof:

- `iqa_dvc_reproducibility`, `iqa_ingestion`, `iqa_replay`,
  `iqa_monitoring`, `iqa_lifecycle` and `iqa_lifecycle_trigger` are listed.
- import errors are empty.
- pool `iqa_gpu` exists.

Trigger the DVC gate:

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.prod.yml exec airflow-webserver \
  airflow dags trigger iqa_dvc_reproducibility \
  --conf '{"with_network": false, "skip_regeneration": true}'
```

The run must finish `success`. DVC remains a reproducibility gate; business DAGs
do not run `dvc push` automatically.

## 4. Model And Data Lineage Evidence

Restore model artifacts from MinIO with strict checksums:

```bash
set -a
source .env
set +a
export IQA_S3_ENDPOINT_URL=http://localhost:9000

uv run --extra cu128 iqa-restore-model-artifacts \
  --model-version roi_segmenter_v001_fixed \
  --strict-checksum

uv run --extra cu128 iqa-restore-model-artifacts \
  --model-version rd_feature_ae_gated_v001_bootstrap \
  --strict-checksum
```

Run a decision-only replay lifecycle proof:

```bash
uv run --extra cu128 iqa-run-replay-lifecycle-cycle \
  --scenario-id production_replay_natural \
  --image-root /opt/iqa/iqa-mlops/data/raw/hss-iad \
  --mode decision-only \
  --max-events 60 \
  --wait-for-gpu
```

Build the compact lineage summary:

```bash
uv run --extra cu128 iqa-lineage-summary \
  --replay-run-dir .cache/iqa/replay_lifecycle/production_replay_natural/<run_id> \
  --model-version rd_feature_ae_gated_v001_bootstrap
```

The summary links lots, scenario, dataset versions, DVC stages, model manifest,
MinIO checkpoint URI, SHA256, preprocessing contract and threshold source.

## 5. MLflow Registry Evidence

For a full training proof in stage `test` only:

```bash
uv run --extra cu128 iqa-run-replay-lifecycle-cycle \
  --scenario-id production_replay_natural \
  --image-root /opt/iqa/iqa-mlops/data/raw/hss-iad \
  --mode train-on-trigger \
  --max-events 60 \
  --stage test \
  --epochs 1 \
  --max-steps 5 \
  --wait-for-gpu
```

Then require MLflow evidence:

```bash
uv run --extra cu128 iqa-lineage-summary \
  --replay-run-dir .cache/iqa/replay_lifecycle/production_replay_natural/<run_id> \
  --model-version rd_feature_ae_gated_v001_bootstrap \
  --require-mlflow-run
```

Expected proof:

- `mlflow_run_id` is present.
- required tags include `dataset_version`, `manifest_version`, `git_commit`,
  `scenario_id` and `preprocessing_contract_version`.
- `MLflow Registry` is the source of truth for active model governance.
- `MinIO` stores checkpoints and artifacts.
- `PostgreSQL` stores facts, statuses, timestamps, versions, URIs and JSONB
  payloads, not binary artifacts.

## 6. Demo Reading

- **Sophie** sees the quality review surface and prediction/feedback path. In the
  MVP, oracle GT remains the sovereign feedback source for training.
- **Marc** follows lots, scenarios, lifecycle triggers, decision thresholds and
  lineage summaries to explain industrial quality decisions.
- **Laurent** verifies authentication boundaries, auditability, Docker Hub image
  provenance, Airflow container isolation, and separation between API,
  PostgreSQL metadata, DVC/MinIO data artifacts, MinIO checkpoints and MLflow
  Registry governance.

The final message for the soutenance: IQA is not only an ML API. It is a
traceable operating loop where data, models, deployments and governance remain
separated and auditable.
