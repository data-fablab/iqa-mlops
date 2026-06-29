from __future__ import annotations

import json
from pathlib import Path

import torch

from iqa.models.artifacts import FEATURE_AE_REFERENCE_CONTRACT_VERSION, load_feature_ae_runtime_contract
from scripts import backfill_feature_ae_runtime_contracts as backfill


def _write_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": {"weight": torch.zeros(256)}}, path)


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation_set_id": "validation_set_piece_b_full_v001",
                "score_contract": {
                    "score_contract_version": FEATURE_AE_REFERENCE_CONTRACT_VERSION,
                    "layer_score_mode": "sqrt_l2_plus_cosine",
                    "layer_normalization": "good_p99",
                    "layer_normalization_stats": {"layer2": 1.2, "layer3": 2.4},
                    "layer_weights": {"layer2": 0.65, "layer3": 0.35},
                    "cosine_weight": 0.5,
                    "score_smoothing": "median3",
                    "roi_threshold": 0.5,
                    "score_image": "topk_mean",
                    "topk_fraction": 0.005,
                },
                "metrics": {"pixel_aupimo_1e-5_1e-3": 0.2, "image_ap": 0.9},
                "images": [{"image_id": "good_001", "score": 1.0, "is_defective": False}],
            }
        ),
        encoding="utf-8",
    )


def _thresholds(role: str) -> dict[str, object]:
    return {
        "score_contract_version": FEATURE_AE_REFERENCE_CONTRACT_VERSION,
        "threshold_orange": 1.1,
        "threshold_red": 1.2,
        "threshold_source": f"panel_good_quantiles:validation:{role}",
    }


def test_backfill_runtime_contracts_dry_run_writes_promoted_role_contracts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_id = "production_replay_natural_piece_b_full"
    run_dir = tmp_path / ".cache/iqa/replay_lifecycle" / scenario_id / "replay_lifecycle_run"
    cycle_dir = run_dir / "cycles" / "cycle_001"
    classification_checkpoint = tmp_path / ".cache/iqa/models" / scenario_id / "replay_lifecycle_run" / "cycle_001" / "checkpoint_best_image.pt"
    localization_checkpoint = classification_checkpoint.with_name("checkpoint_best_localization.pt")
    classification_metrics = cycle_dir / "evaluation/reference_classification/candidate/metrics.json"
    localization_metrics = cycle_dir / "evaluation/reference_localization/candidate/metrics.json"
    _write_checkpoint(classification_checkpoint)
    _write_checkpoint(localization_checkpoint)
    _write_metrics(classification_metrics)
    _write_metrics(localization_metrics)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cycles.jsonl").write_text(
        json.dumps(
            {
                "cycle_id": "cycle_001",
                "scenario_id": scenario_id,
                "lifecycle_run_id": "replay_lifecycle_run",
                "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
                "mlflow_run_id": "run-001",
                "classification_promotion_status": "promoted",
                "classification_candidate_checkpoint": classification_checkpoint.relative_to(tmp_path).as_posix(),
                "classification_candidate_eval_metrics_path": classification_metrics.relative_to(tmp_path).as_posix(),
                "classification_candidate_decision_thresholds": _thresholds("classification"),
                "classification_registered_model_name": f"feature_ae_classifier__{scenario_id}",
                "classification_registered_model_version": "5",
                "localization_promotion_status": "promoted",
                "localization_candidate_checkpoint": localization_checkpoint.relative_to(tmp_path).as_posix(),
                "localization_candidate_eval_metrics_path": localization_metrics.relative_to(tmp_path).as_posix(),
                "localization_candidate_decision_thresholds": _thresholds("localization"),
                "localization_registered_model_name": f"feature_ae_localization__{scenario_id}",
                "localization_registered_model_version": "9",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = backfill.backfill_runtime_contracts(
        run_dir=run_dir,
        scenario_id=scenario_id,
        repo_root=tmp_path,
        dry_run=True,
    )

    assert result["status_counts"] == {"dry_run": 2}
    classification_contract = cycle_dir / "runtime_contracts/classification/runtime_contract.json"
    localization_contract = cycle_dir / "runtime_contracts/localization/runtime_contract.json"
    assert classification_contract.exists()
    assert localization_contract.exists()
    payload = load_feature_ae_runtime_contract(classification_contract)
    assert payload["decision_thresholds"]["threshold_red"] == 1.2
    assert payload["feature_ae_reference_contract"]["layer_normalization_stats"] == {"layer2": 1.2, "layer3": 2.4}


def test_backfill_runtime_contracts_logs_to_existing_run_without_creating_registry_version(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scenario_id = "production_replay_natural_piece_b_full"
    run_dir = tmp_path / ".cache/iqa/replay_lifecycle" / scenario_id / "replay_lifecycle_run"
    cycle_dir = run_dir / "cycles" / "cycle_001"
    checkpoint = tmp_path / ".cache/iqa/models" / scenario_id / "replay_lifecycle_run" / "cycle_001" / "checkpoint_best_image.pt"
    metrics = cycle_dir / "evaluation/reference_classification/candidate/metrics.json"
    _write_checkpoint(checkpoint)
    _write_metrics(metrics)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cycles.jsonl").write_text(
        json.dumps(
            {
                "cycle_id": "cycle_001",
                "scenario_id": scenario_id,
                "lifecycle_run_id": "replay_lifecycle_run",
                "candidate_version": "rd_feature_ae_gated_natural_cycle_001",
                "mlflow_run_id": "run-001",
                "classification_promotion_status": "promoted",
                "classification_candidate_checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
                "classification_candidate_eval_metrics_path": metrics.relative_to(tmp_path).as_posix(),
                "classification_candidate_decision_thresholds": _thresholds("classification"),
                "classification_registered_model_name": f"feature_ae_classifier__{scenario_id}",
                "classification_registered_model_version": "5",
                "localization_promotion_status": "rejected_reference_regression",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: dict[str, list[tuple[object, ...]]] = {"artifacts": [], "run_tags": [], "version_tags": []}

    class FakeClient:
        def log_artifact(self, run_id: str, local_path: str, artifact_path: str) -> None:
            calls["artifacts"].append((run_id, Path(local_path).name, artifact_path))

        def set_tag(self, run_id: str, key: str, value: str) -> None:
            calls["run_tags"].append((run_id, key, value))

        def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> None:
            calls["version_tags"].append((name, version, key, value))

        def create_model_version(self, *args, **kwargs):  # pragma: no cover - must never be called.
            raise AssertionError("backfill must not create registry versions")

    monkeypatch.setattr(backfill, "_mlflow_client", lambda tracking_uri: FakeClient())

    result = backfill.backfill_runtime_contracts(
        run_dir=run_dir,
        scenario_id=scenario_id,
        repo_root=tmp_path,
        dry_run=False,
    )

    assert result["status_counts"] == {"backfilled": 1, "skipped": 1}
    assert ("run-001", "runtime_contract.json", "runtime_contracts/classification") in calls["artifacts"]
    assert ("run-001", "classification_runtime_contract_backfilled", "true") in calls["run_tags"]
    assert (
        f"feature_ae_classifier__{scenario_id}",
        "5",
        "model_role",
        "classification",
    ) in calls["version_tags"]
