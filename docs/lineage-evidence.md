# Phase 3 Lineage Evidence

This document is the operator-facing proof that IQA can trace a Phase 3 model
decision from data replay to model runtime artifacts without mixing metadata,
datasets and binary checkpoints.

## Target Chain

The evidence chain is:

```text
piece_event_id -> lot_id -> scenario_id -> dataset_version -> manifest_version
-> DVC/MinIO snapshot -> model_version -> MLflow run -> model manifest -> registry
```

The replay lifecycle runner materializes the runtime side of this chain in:

```text
.cache/iqa/replay_lifecycle/<scenario_id>/<run_id>/events.jsonl
.cache/iqa/replay_lifecycle/<scenario_id>/<run_id>/lots.jsonl
.cache/iqa/replay_lifecycle/<scenario_id>/<run_id>/summary.json
```

The model side is stored as lightweight Git manifests under
`models/manifests/<model_version>/model_manifest.json`. Heavy checkpoints stay
in MinIO and are restored into `.cache/iqa/models/`.

## Storage Boundaries

| Layer | Responsibility |
| --- | --- |
| Git | Code, tests, docs, configs, contracts and lightweight manifests |
| DVC/MinIO | Versioned data artifacts and dataset snapshots in `s3://iqa-dvc` |
| Model MinIO | Checkpoints in `s3://iqa-models` |
| MLflow | Tracking, metrics, artifacts and Registry source of truth |
| PostgreSQL metadata | Facts, statuses, timestamps, versions, URIs and JSONB payloads |

PostgreSQL never stores checkpoints, masks, heatmaps or image binaries. DVC
remains a reproducibility and data lineage gate; business DAGs do not run
`dvc push` automatically.

## MLflow Evidence

Feature-AE training must carry enough tags and params to reconnect the run to
the replay/data context:

- `dataset_version`
- `manifest_version`
- `git_commit`
- `scenario_id`
- `model_version` or candidate version
- `preprocessing_contract_version`

The MLflow Registry remains the source of truth for active/promoted model
versions. MinIO stores the files; MLflow decides which model is active.

## Operator Summary Command

After a replay lifecycle run, generate a compact proof document:

```bash
uv run --extra cpu iqa-lineage-summary \
  --replay-run-dir .cache/iqa/replay_lifecycle/production_replay_natural/<run_id> \
  --model-version rd_feature_ae_gated_v001_bootstrap
```

Optional output file:

```bash
uv run --extra cpu iqa-lineage-summary \
  --replay-run-dir .cache/iqa/replay_lifecycle/production_replay_natural/<run_id> \
  --model-version rd_feature_ae_gated_v001_bootstrap \
  --output .cache/iqa/lineage/production_replay_natural_summary.json
```

The summary includes the replay scenario, processed lots, dataset versions,
model artifact URI, SHA256, preprocessing contract, decision thresholds, DVC
stages and MLflow traceability fields.

## Server Evidence Flow

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

Run a calibrated replay lifecycle proof:

```bash
uv run --extra cu128 iqa-run-replay-lifecycle-cycle \
  --scenario-id production_replay_natural \
  --image-root /opt/iqa/iqa-mlops/data/raw/hss-iad \
  --mode decision-only \
  --max-events 60 \
  --wait-for-gpu
```

Build the lineage evidence:

```bash
uv run --extra cu128 iqa-lineage-summary \
  --replay-run-dir .cache/iqa/replay_lifecycle/production_replay_natural/<run_id> \
  --model-version rd_feature_ae_gated_v001_bootstrap
```

Validate DVC/MinIO explicitly:

```bash
uv run --extra cpu --extra data iqa-check-dvc-reproducibility --with-network
```

This last command is an operator/CI gate. It is not a business trigger and does
not replace the lifecycle decision rules.

## Demo Reading

For Marc, the evidence shows which lots and scenarios produced the decision and
which oracle/data versions were used.

For Laurent, the evidence shows separation of duties: API and metadata carry
facts and URIs, DVC/MinIO carry data artifacts, MinIO carries checkpoints, and
MLflow Registry remains the source of truth for active model governance.
