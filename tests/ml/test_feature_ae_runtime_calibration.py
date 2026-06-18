from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import torch
from PIL import Image

from iqa.inference.feature_ae import score_feature_ae_map
from scripts.calibrate_feature_ae_thresholds import build_decision_thresholds


def test_roi_topk_score_uses_valid_mask(tmp_path: Path) -> None:
    score_map = torch.tensor(
        [
            [0.1, 0.2, 9.0],
            [0.3, 4.0, 8.0],
            [0.4, 0.5, 7.0],
        ],
        dtype=torch.float32,
    )
    mask = Image.new("L", (3, 3), 0)
    mask.putpixel((0, 0), 255)
    mask.putpixel((1, 0), 255)
    mask.putpixel((0, 1), 255)
    mask_path = tmp_path / "roi.png"
    mask.save(mask_path)

    score = score_feature_ae_map(
        score_map,
        roi_mask_path=mask_path,
        score_smoothing="none",
        score_image="topk_mean",
        topk_fraction=0.30,
    )

    assert score == pytest.approx(0.3)


def test_threshold_quantiles_are_deterministic_and_loss_free() -> None:
    thresholds = build_decision_thresholds(
        [0.0, 1.0, 2.0, 3.0],
        model_version="rd_feature_ae_gated_v001_bootstrap",
        calibration_set_id="calibration_set_v001",
        orange_quantile=0.5,
        red_quantile=0.75,
    )

    assert thresholds["threshold_orange"] == pytest.approx(1.5)
    assert thresholds["threshold_red"] == pytest.approx(2.25)
    assert "val_loss" not in thresholds
    assert thresholds["sample_count"] == 4


def test_threshold_quantiles_reject_empty_scores() -> None:
    with pytest.raises(ValueError, match="without scores"):
        build_decision_thresholds(
            [],
            model_version="rd_feature_ae_gated_v001_bootstrap",
            calibration_set_id="calibration_set_v001",
            orange_quantile=0.95,
            red_quantile=0.99,
        )


def test_calibration_cli_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["iqa-calibrate-feature-ae-thresholds"]
        == "scripts.calibrate_feature_ae_thresholds:main"
    )
