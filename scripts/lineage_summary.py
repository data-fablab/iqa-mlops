"""Build a compact Phase 3 data/model lineage evidence summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from iqa.models.artifacts import DEFAULT_FEATURE_AE_MODEL_VERSION, load_model_manifest, model_manifest_path
from iqa.registry import registered_model_name

DEFAULT_DVC_YAML = Path("dvc.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run-dir", type=Path, required=True)
    parser.add_argument("--model-version", default=DEFAULT_FEATURE_AE_MODEL_VERSION)
    parser.add_argument("--dvc-yaml", type=Path, default=DEFAULT_DVC_YAML)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-mlflow-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_lineage_summary(
        replay_run_dir=args.replay_run_dir,
        model_version=args.model_version,
        dvc_yaml=args.dvc_yaml,
        require_mlflow_run=args.require_mlflow_run,
    )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


def build_lineage_summary(
    *,
    replay_run_dir: Path,
    model_version: str,
    dvc_yaml: Path = DEFAULT_DVC_YAML,
    require_mlflow_run: bool = False,
) -> dict[str, Any]:
    run_summary = _read_required_json(replay_run_dir / "summary.json")
    events = _read_jsonl(replay_run_dir / "events.jsonl")
    lots = _read_jsonl(replay_run_dir / "lots.jsonl")
    manifest = load_model_manifest(model_version)
    dvc_stages = _read_dvc_stages(dvc_yaml)

    dataset_versions = _stable_unique(
        [
            *[str(event.get("dataset_version") or "") for event in events],
            str(run_summary.get("candidate_dataset_version") or ""),
            str(manifest.get("dataset_version") or ""),
        ]
    )
    lot_ids = _stable_unique(str(lot.get("lot_id") or "") for lot in lots)
    source_classes = _stable_unique(str(event.get("source_class") or "") for event in events)
    threshold_sources = _stable_unique(str(event.get("threshold_source") or "") for event in events)

    mlflow_run_id = str(run_summary.get("mlflow_run_id") or "")
    if require_mlflow_run and not mlflow_run_id:
        raise ValueError("lineage summary requires mlflow_run_id, but summary.json has no MLflow run.")
    scenario_id = str(run_summary.get("scenario_id") or "")
    registry_model_name = registered_model_name(scenario_id) if scenario_id else None
    summary = {
        "scenario_id": scenario_id or None,
        "run_id": run_summary.get("run_id"),
        "mode": run_summary.get("mode"),
        "events_processed": run_summary.get("events_processed"),
        "lots_processed": run_summary.get("lots_processed"),
        "trigger_lifecycle": run_summary.get("trigger_lifecycle"),
        "trigger_reason": run_summary.get("trigger_reason"),
        "candidate_dataset_version": run_summary.get("candidate_dataset_version"),
        "candidate_checkpoint": run_summary.get("candidate_checkpoint"),
        "mlflow_run_id": mlflow_run_id or None,
        "mlflow_tracking": {
            "source_of_truth": "mlflow_registry",
            "run_id": mlflow_run_id or None,
            "evidence_status": "present" if mlflow_run_id else "absent_decision_only",
            "registered_model_name": registry_model_name,
            "registry_source_of_truth": "mlflow_registry",
            "required_tags": [
                "dataset_version",
                "manifest_version",
                "git_commit",
                "scenario_id",
                "model_version",
                "candidate_version",
                "roi_model_version",
                "feature_ae_version",
                "preprocessing_contract_version",
            ],
        },
        "dataset_versions": dataset_versions,
        "lot_ids": lot_ids,
        "source_classes": source_classes,
        "event_count_from_jsonl": len(events),
        "lot_count_from_jsonl": len(lots),
        "model_version": manifest.get("model_version") or model_version,
        "model_manifest": str(model_manifest_path(model_version)),
        "model_artifact_uri": manifest.get("artifact_uri"),
        "model_sha256": manifest.get("sha256"),
        "model_dataset_version": manifest.get("dataset_version"),
        "preprocessing_contract_version": manifest.get("preprocessing_contract_version"),
        "roi_model_version": manifest.get("roi_model_version"),
        "decision_thresholds": manifest.get("decision_thresholds"),
        "threshold_sources": threshold_sources,
        "dvc": {
            "remote": "s3://iqa-dvc",
            "gate_command": "iqa-check-dvc-reproducibility",
            "stages": dvc_stages,
        },
        "storage_boundaries": {
            "git": "code, tests, docs, configs, contracts and lightweight manifests",
            "dvc_minio": "versioned data artifacts and dataset snapshots",
            "model_minio": "model checkpoints under s3://iqa-models",
            "mlflow": "tracking, registry and ML artifacts",
            "postgresql": "facts, statuses, timestamps, versions, URIs and JSONB payloads",
        },
    }
    summary["lineage_status"] = _lineage_status(summary)
    return summary


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing lineage input file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"lineage input is not a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing lineage input file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"lineage JSONL row is not an object: {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"lineage JSONL file is empty: {path}")
    return rows


def _read_dvc_stages(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing DVC pipeline file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    stages = payload.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError(f"DVC pipeline has no stages mapping: {path}")
    return sorted(str(stage) for stage in stages)


def _stable_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _lineage_status(summary: dict[str, Any]) -> str:
    required = [
        summary.get("scenario_id"),
        summary.get("dataset_versions"),
        summary.get("lot_ids"),
        summary.get("model_artifact_uri"),
        summary.get("model_sha256"),
        summary.get("preprocessing_contract_version"),
        summary.get("dvc", {}).get("stages"),
    ]
    return "complete" if all(required) else "incomplete"


if __name__ == "__main__":
    main()
