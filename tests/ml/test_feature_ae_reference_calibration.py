from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import pytest

from iqa.models.feature_ae import REFERENCE_FEATURE_AE_CONTRACT
from scripts.calibrate_feature_ae_reference import (
    assert_validation_has_defects,
    build_reference_thresholds,
    select_business_metric,
    update_reference_manifest,
    write_calibration_matrix,
)


def test_reference_business_metric_priority_prefers_pixel_aupimo() -> None:
    metric, value = select_business_metric(
        {
            "image_ap": 0.91,
            "pixel_ap": 0.44,
            "pixel_aupimo_1e-5_1e-3": 0.417,
        }
    )

    assert metric == "pixel_aupimo_1e-5_1e-3"
    assert value == pytest.approx(0.417)


def test_reference_business_metric_falls_back_to_pixel_ap() -> None:
    metric, value = select_business_metric({"image_ap": 0.91, "pixel_ap": 0.44})

    assert metric == "pixel_ap"
    assert value == pytest.approx(0.44)


def test_reference_thresholds_expose_scoring_contract() -> None:
    thresholds = build_reference_thresholds(
        [1.0, 2.0, 3.0, 4.0],
        model_version="candidate",
        validation_manifest=Path("data/validation/validation_set_v001.csv"),
        gt_masks_manifest=Path("data/validation/validation_gt_masks_v001.csv"),
        layer_weights={"layer2": 0.65, "layer3": 0.35},
        orange_quantile=0.5,
        red_quantile=0.75,
        selected_metric="pixel_aupimo_1e-5_1e-3",
        selected_metric_value=0.417,
    )

    assert thresholds["score_contract_version"] == REFERENCE_FEATURE_AE_CONTRACT.version
    assert thresholds["teacher_weights"] == "IMAGENET1K_V1"
    assert thresholds["layer_weights"] == {"layer2": 0.65, "layer3": 0.35}
    assert thresholds["roi_mode"] == "soft_map"
    assert thresholds["roi_threshold"] == pytest.approx(0.5)
    assert thresholds["topk_fraction"] == pytest.approx(0.005)
    assert thresholds["selected_metric"] == "pixel_aupimo_1e-5_1e-3"
    assert thresholds["threshold_orange"] == pytest.approx(2.5)
    assert thresholds["threshold_red"] == pytest.approx(3.25)
    assert "val_loss" not in thresholds


def test_reference_validation_requires_defective_gt_masks(tmp_path: Path) -> None:
    validation = tmp_path / "validation.csv"
    gt_masks = tmp_path / "gt.csv"
    validation.write_text("is_defective,label\nFalse,good\n", encoding="utf-8")
    gt_masks.write_text("image_id,gt_mask_path\n", encoding="utf-8")

    with pytest.raises(ValueError, match="defective validation images"):
        assert_validation_has_defects(validation, gt_masks)


def test_reference_calibration_matrix_contains_pixel_metrics(tmp_path: Path) -> None:
    path = write_calibration_matrix(
        tmp_path / "calibration_matrix.csv",
        {
            "pixel_aupimo_1e-5_1e-3": 0.417,
            "pixel_ap": 0.377,
            "image_ap": 0.922,
            "image_auroc": 0.868,
        },
        {
            "score_contract_version": "feature_ae_reference_v001",
            "selected_metric": "pixel_aupimo_1e-5_1e-3",
            "selected_metric_value": 0.417,
            "threshold_orange": 12.0,
            "threshold_red": 20.0,
        },
    )

    with path.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["pixel_aupimo_1e-5_1e-3"] == "0.417"
    assert row["pixel_ap"] == "0.377"
    assert row["selected_metric"] == "pixel_aupimo_1e-5_1e-3"


def test_update_reference_manifest_writes_complete_contract(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text('{"model_version": "candidate"}\n', encoding="utf-8")
    monkeypatch.setattr("scripts.calibrate_feature_ae_reference.model_manifest_path", lambda _version: manifest_path)

    update_reference_manifest(
        "candidate",
        {
            "score_contract_version": "feature_ae_reference_v001",
            "threshold_orange": 12.0,
            "threshold_red": 20.0,
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["feature_ae_reference_contract"]
    assert manifest["preprocessing_contract_version"] == "feature_ae_reference_v001"
    assert contract["teacher_weights"] == "IMAGENET1K_V1"
    assert contract["layer_weights"] == {"layer2": 0.65, "layer3": 0.35}
    assert contract["tile_size"] == 384
    assert contract["context_size"] == 768
    assert manifest["decision_thresholds"]["threshold_red"] == 20.0


def test_reference_calibration_cli_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["iqa-calibrate-feature-ae-reference"]
        == "scripts.calibrate_feature_ae_reference:main"
    )

