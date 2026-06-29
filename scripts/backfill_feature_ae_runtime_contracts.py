"""Backfill Feature-AE runtime contracts for already promoted lifecycle runs.

The backfill is intentionally narrow: it does not create registry versions and
does not re-evaluate models. It reconstructs the runtime contract from local
gate evaluation artifacts and attaches it to the existing MLflow source run.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqa.models.artifacts import load_feature_ae_runtime_contract
from iqa.storage.artifacts import sha256_file
from scripts.run_replay_lifecycle_cycle import (
    CLASSIFICATION_MODEL_NAME_BASE,
    LOCALIZATION_MODEL_NAME_BASE,
    registered_model_name,
    safe_checkpoint_reference,
    write_feature_ae_runtime_contract,
)


@dataclass(frozen=True)
class BackfillEntry:
    cycle_id: str
    role: str
    status: str
    reason: str
    mlflow_run_id: str = ""
    registered_model_name: str = ""
    registered_model_version: str = ""
    runtime_contract_path: str = ""
    runtime_contract_sha256: str = ""
    checkpoint_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "role": self.role,
            "status": self.status,
            "reason": self.reason,
            "mlflow_run_id": self.mlflow_run_id,
            "registered_model_name": self.registered_model_name,
            "registered_model_version": self.registered_model_version,
            "runtime_contract_path": self.runtime_contract_path,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Feature-AE runtime_contract artifacts on existing promoted MLflow runs."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Lifecycle output directory containing cycles.jsonl.")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI") or os.getenv("IQA_MLFLOW_TRACKING_URI") or "")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target-stage", default="test")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = backfill_runtime_contracts(
        run_dir=args.run_dir,
        scenario_id=args.scenario_id,
        repo_root=args.repo_root,
        tracking_uri=args.tracking_uri,
        target_stage=args.target_stage,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def backfill_runtime_contracts(
    *,
    run_dir: Path,
    scenario_id: str,
    repo_root: Path,
    tracking_uri: str = "",
    target_stage: str = "test",
    dry_run: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    repo_root = repo_root.resolve()
    cycles = _read_cycles(run_dir / "cycles.jsonl")
    client = None if dry_run else _mlflow_client(tracking_uri)
    entries: list[BackfillEntry] = []
    for cycle in cycles:
        for role in ("classification", "localization"):
            entries.append(
                _backfill_role(
                    cycle,
                    role=role,
                    run_dir=run_dir,
                    repo_root=repo_root,
                    scenario_id=scenario_id,
                    target_stage=target_stage,
                    client=client,
                    dry_run=dry_run,
                )
            )
    status_counts: dict[str, int] = {}
    for entry in entries:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
    return {
        "status": "validated",
        "dry_run": dry_run,
        "run_dir": str(run_dir),
        "scenario_id": scenario_id,
        "cycle_count": len(cycles),
        "entries": [entry.to_dict() for entry in entries],
        "status_counts": status_counts,
    }


def _read_cycles(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"cycles.jsonl is missing: {path}")
    cycles: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cycles.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
    return cycles


def _backfill_role(
    cycle: dict[str, Any],
    *,
    role: str,
    run_dir: Path,
    repo_root: Path,
    scenario_id: str,
    target_stage: str,
    client: Any,
    dry_run: bool,
) -> BackfillEntry:
    cycle_id = str(cycle.get("cycle_id") or "")
    promoted = cycle.get(f"{role}_promotion_status") == "promoted"
    if not promoted:
        return BackfillEntry(cycle_id=cycle_id, role=role, status="skipped", reason="role_not_promoted")

    mlflow_run_id = str(cycle.get("mlflow_run_id") or "")
    if not mlflow_run_id:
        return BackfillEntry(cycle_id=cycle_id, role=role, status="skipped", reason="missing_mlflow_run_id")

    checkpoint_path = _resolve_existing_path(
        _role_value(cycle, role, "checkpoint"),
        repo_root=repo_root,
        run_dir=run_dir,
    )
    metrics_path = _resolve_existing_path(
        _role_value(cycle, role, "metrics_path"),
        repo_root=repo_root,
        run_dir=run_dir,
    )
    if checkpoint_path is None:
        return BackfillEntry(
            cycle_id=cycle_id,
            role=role,
            status="failed",
            reason="missing_checkpoint",
            mlflow_run_id=mlflow_run_id,
        )
    if metrics_path is None:
        return BackfillEntry(
            cycle_id=cycle_id,
            role=role,
            status="failed",
            reason="missing_candidate_metrics",
            mlflow_run_id=mlflow_run_id,
        )

    cycle_dir = run_dir / "cycles" / cycle_id
    thresholds = _role_value(cycle, role, "thresholds")
    contract_path = write_feature_ae_runtime_contract(
        output_path=cycle_dir / "runtime_contracts" / role / "runtime_contract.json",
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        model_version=str(cycle.get("candidate_version") or ""),
        model_role=role,
        decision_thresholds=thresholds if isinstance(thresholds, dict) else None,
        scenario_id=scenario_id,
        lifecycle_run_id=str(cycle.get("lifecycle_run_id") or cycle.get("run_id") or run_dir.name),
        cycle_id=cycle_id,
    )
    if not contract_path:
        return BackfillEntry(
            cycle_id=cycle_id,
            role=role,
            status="failed",
            reason="runtime_contract_not_written",
            mlflow_run_id=mlflow_run_id,
        )
    load_feature_ae_runtime_contract(contract_path)
    contract_sha = sha256_file(Path(contract_path))
    checkpoint_sha = sha256_file(checkpoint_path)
    model_name = str(cycle.get(f"{role}_registered_model_name") or "") or registered_model_name(
        scenario_id,
        base_name=CLASSIFICATION_MODEL_NAME_BASE if role == "classification" else LOCALIZATION_MODEL_NAME_BASE,
    )
    model_version = str(cycle.get(f"{role}_registered_model_version") or "")

    if not dry_run:
        _log_runtime_contract_to_mlflow(
            client,
            run_id=mlflow_run_id,
            role=role,
            contract_path=Path(contract_path),
            contract_sha=contract_sha,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
            cycle=cycle,
            model_name=model_name,
            model_version=model_version,
            target_stage=target_stage,
        )

    return BackfillEntry(
        cycle_id=cycle_id,
        role=role,
        status="dry_run" if dry_run else "backfilled",
        reason="ok",
        mlflow_run_id=mlflow_run_id,
        registered_model_name=model_name,
        registered_model_version=model_version,
        runtime_contract_path=safe_checkpoint_reference(Path(contract_path)),
        runtime_contract_sha256=contract_sha,
        checkpoint_sha256=checkpoint_sha,
    )


def _role_value(cycle: dict[str, Any], role: str, kind: str) -> Any:
    if role == "classification":
        if kind == "checkpoint":
            return cycle.get("classification_candidate_checkpoint") or cycle.get("candidate_checkpoint")
        if kind == "metrics_path":
            return cycle.get("classification_candidate_eval_metrics_path") or cycle.get("candidate_eval_metrics_path")
        if kind == "thresholds":
            return cycle.get("classification_candidate_decision_thresholds") or cycle.get("candidate_decision_thresholds")
    if role == "localization":
        if kind == "checkpoint":
            return cycle.get("localization_candidate_checkpoint") or cycle.get("candidate_checkpoint")
        if kind == "metrics_path":
            return cycle.get("localization_candidate_eval_metrics_path") or cycle.get("candidate_eval_metrics_path")
        if kind == "thresholds":
            return cycle.get("localization_candidate_decision_thresholds") or cycle.get("candidate_decision_thresholds")
    return None


def _resolve_existing_path(value: Any, *, repo_root: Path, run_dir: Path) -> Path | None:
    if not value:
        return None
    raw = str(value).replace("\\", "/")
    candidates: list[Path] = []
    path = Path(str(value))
    if path.is_absolute():
        candidates.append(path)
    if raw.startswith("/app/"):
        candidates.append(repo_root / raw.removeprefix("/app/"))
    if "/iqa-mlops/" in raw:
        candidates.append(repo_root / raw.split("/iqa-mlops/", 1)[1])
    candidates.append(repo_root / raw)
    candidates.append(run_dir / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _mlflow_client(tracking_uri: str) -> Any:
    import mlflow

    resolved_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or os.getenv("IQA_MLFLOW_TRACKING_URI")
    if resolved_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", resolved_tracking_uri)
        mlflow.set_tracking_uri(resolved_tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=resolved_tracking_uri or None)


def _log_runtime_contract_to_mlflow(
    client: Any,
    *,
    run_id: str,
    role: str,
    contract_path: Path,
    contract_sha: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    cycle: dict[str, Any],
    model_name: str,
    model_version: str,
    target_stage: str,
) -> None:
    client.log_artifact(run_id, str(contract_path), artifact_path=f"runtime_contracts/{role}")
    timestamp = datetime.now(UTC).isoformat()
    run_tags = {
        f"{role}_runtime_contract_backfilled": "true",
        f"{role}_runtime_contract_sha256": contract_sha,
        f"{role}_runtime_contract_path": safe_checkpoint_reference(contract_path),
        f"{role}_checkpoint_sha256": checkpoint_sha,
        "runtime_contract_backfilled_at": timestamp,
    }
    for key, value in run_tags.items():
        client.set_tag(run_id, key, value)
    if not model_name or not model_version:
        return
    version_tags = {
        "model_role": role,
        "scenario_id": str(cycle.get("scenario_id") or ""),
        "lifecycle_run_id": str(cycle.get("lifecycle_run_id") or cycle.get("run_id") or ""),
        "cycle_id": str(cycle.get("cycle_id") or ""),
        "candidate_version": str(cycle.get("candidate_version") or ""),
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "runtime_contract_sha256": contract_sha,
        "runtime_contract_backfilled": "true",
        "runtime_contract_backfilled_at": timestamp,
        "registry_alias": target_stage,
    }
    for key, value in version_tags.items():
        if value:
            client.set_model_version_tag(model_name, model_version, key, value)


if __name__ == "__main__":
    main()
