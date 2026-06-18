from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import numpy as np

from iqa.storage.artifacts import sha256_file
from iqa.training.bootstrap import (
    BOOTSTRAP_ARTIFACT_URI,
    materialize_bootstrap_checkpoint,
    select_bootstrap_champion,
    update_bootstrap_manifest,
    upload_checkpoint_to_s3,
)
from iqa.training.feature_ae import FeatureAETrainingConfig, train_feature_ae
from iqa.training.feature_ae_contracts import FEATURE_AE_PREPROCESSING_CONTRACT_VERSION
from iqa.training.feature_ae_evaluation import compute_binary_metrics
from scripts import build_feature_ae_bootstrap


def _checkpoint(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _write_metric_best(run_dir: Path, payload: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metric_eval_best.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_loss_history(run_dir: Path) -> None:
    (run_dir / "loss_history.csv").write_text(
        "epoch,train_loss,val_loss,lr\n"
        "1,0.2,0.1,0.001\n"
        "2,0.3,0.9,0.001\n",
        encoding="utf-8",
    )


def test_bootstrap_selection_prioritizes_business_metrics_over_val_loss(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _checkpoint(run_dir / "checkpoint_best_image_ap.pt", b"image-ap")
    _checkpoint(run_dir / "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt", b"aupimo")
    _write_metric_best(
        run_dir,
        {
            "image_ap": {"value": 0.99, "epoch": 1, "checkpoint": "checkpoint_best_image_ap.pt"},
            "pixel_aupimo_1e-5_1e-3": {
                "value": 0.50,
                "epoch": 2,
                "checkpoint": "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
            },
        },
    )
    _write_loss_history(run_dir)

    champion = select_bootstrap_champion(run_dir)

    assert champion.selected_metric == "pixel_aupimo_1e-5_1e-3"
    assert champion.selected_epoch == 2
    assert champion.val_loss == 0.9


def test_bootstrap_selection_fails_without_business_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metric_best(run_dir, {})

    with pytest.raises(ValueError, match="No bootstrap business metric"):
        select_bootstrap_champion(run_dir)


def test_materialize_and_update_bootstrap_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source = _checkpoint(run_dir / "checkpoint_best_pixel_ap.pt", b"pixel-ap")
    _write_metric_best(run_dir, {"pixel_ap": {"value": 0.7, "epoch": 3, "checkpoint": source.name}})
    champion = select_bootstrap_champion(run_dir)
    manifest = tmp_path / "model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_version": "rd_feature_ae_gated_v001_bootstrap",
                "artifact_uri": BOOTSTRAP_ARTIFACT_URI,
                "sha256": "pending-restore",
            }
        ),
        encoding="utf-8",
    )

    canonical = materialize_bootstrap_checkpoint(champion, run_dir / "checkpoint.pt")
    payload = update_bootstrap_manifest(manifest, champion)

    assert canonical.read_bytes() == b"pixel-ap"
    assert payload["sha256"] == sha256_file(source)
    assert payload["selected_metric"] == "pixel_ap"
    assert payload["selected_metric_value"] == 0.7
    assert payload["selected_epoch"] == 3
    assert payload["dataset_version"] == "feature_ae_good_v001_bootstrap"
    assert payload["validation_set_id"] == "validation_set_v001"
    assert payload["preprocessing_contract_version"] == FEATURE_AE_PREPROCESSING_CONTRACT_VERSION
    assert payload["preprocessing_contract"]["image_size"] == 384


def test_upload_checkpoint_to_s3_uses_manifest_uri(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", b"checkpoint")
    calls: list[tuple[str, str, str]] = []

    class FakeS3Client:
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            calls.append((filename, bucket, key))

    upload_checkpoint_to_s3(checkpoint, BOOTSTRAP_ARTIFACT_URI, s3_client=FakeS3Client())

    assert calls == [(str(checkpoint), "iqa-models", "rd_feature_ae_gated_v001_bootstrap/checkpoint.pt")]


def test_bootstrap_command_dry_run_does_not_require_image_root(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["iqa-build-feature-ae-bootstrap", "--dry-run", "--publish-minio"])

    build_feature_ae_bootstrap.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["publish_minio"] is True
    assert payload["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert payload["preprocessing_contract_version"] == FEATURE_AE_PREPROCESSING_CONTRACT_VERSION
    assert payload["preprocessing_contract"]["augmentation_profile"] == "none"
    assert "pixel_aupimo" in payload["ranking_policy"]


def test_feature_ae_metric_eval_reports_pixel_auroc() -> None:
    metrics = compute_binary_metrics(
        image_labels=[False, True],
        image_scores=[0.1, 0.9],
        pixel_labels=[np.array([[0, 0], [0, 1]], dtype=np.uint8)],
        pixel_scores=[np.array([[0.1, 0.2], [0.3, 0.9]], dtype=np.float32)],
    )

    assert metrics["pixel_auroc"] == 1.0


def test_training_rejects_noncanonical_preprocessing_without_dev_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Non-canonical Feature-AE preprocessing"):
        train_feature_ae(
            FeatureAETrainingConfig(
                manifest_path=tmp_path / "missing.csv",
                image_root=tmp_path,
                output_checkpoint=tmp_path / "checkpoint.pt",
                image_size=32,
            )
        )
