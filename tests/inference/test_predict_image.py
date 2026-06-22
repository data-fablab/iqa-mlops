"""Tests for predict_image functionality with score, roi_status, heatmap, and latency_ms."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from iqa.inference import FeatureAEPrediction, predict_feature_ae_image
from iqa.inference.feature_ae import normalize_feature_ae_display_map, save_feature_ae_heatmap_overlay
from iqa.models.feature_ae import (
    apply_champion_roi,
    fuse_numpy_layer_maps,
    score_numpy_map_topk,
)


def _predict(image_path: Path, checkpoint_path: Path) -> FeatureAEPrediction:
    """Run a single-image Feature-AE prediction with the test tile/context sizes."""
    return predict_feature_ae_image(
        image_path,
        checkpoint_path,
        image_size=32,
        context_size=64,
    )


class TestPredictImage:
    """Test predict_image functionality respecting IQA1_KEN04 requirements."""

    def test_predict_image_returns_score(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """predict_image returns score field."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert hasattr(prediction, "score")
        assert isinstance(prediction.score, float)
        assert prediction.score >= 0.0

    def test_predict_image_returns_roi_status(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """predict_image returns roi_status field (can be None)."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert hasattr(prediction, "roi_status")
        assert prediction.roi_status is None or isinstance(prediction.roi_status, str)

    def test_predict_image_returns_heatmap_uri(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """predict_image returns heatmap_uri field (placeholder or None)."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert hasattr(prediction, "heatmap_uri")
        assert prediction.heatmap_uri is None or isinstance(prediction.heatmap_uri, str)

    def test_predict_image_returns_latency_ms(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """predict_image returns latency_ms field."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert hasattr(prediction, "latency_ms")
        assert isinstance(prediction.latency_ms, float)
        assert prediction.latency_ms > 0.0

    def test_predict_image_latency_measured_on_inference(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """latency_ms is measured during model inference."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert prediction.latency_ms > 0.0
        assert prediction.latency_ms < 10000.0

    def test_predict_image_respects_contract(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """predict_image output respects IQA1_KEN01 model contract."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert isinstance(prediction, FeatureAEPrediction)
        assert prediction.model_type == "reverse_distill_resnet18_dual_context_gated"
        assert prediction.status in {"green", "orange", "red"}
        assert prediction.score >= 0.0
        assert prediction.latency_ms > 0.0

    def test_predict_image_to_dict_includes_all_fields(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """predict_image to_dict includes score, roi_status, heatmap_uri, latency_ms."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        result_dict = prediction.to_dict()
        assert "score" in result_dict
        assert "roi_status" in result_dict
        assert "heatmap_uri" in result_dict
        assert "latency_ms" in result_dict
        assert result_dict["score_contract_version"] == "feature_ae_champion_v001"

    def test_predict_image_latency_ms_is_positive(
        self, sample_image: Path, synthetic_feature_ae_checkpoint: Path
    ) -> None:
        """latency_ms is a positive number representing inference time."""
        prediction = _predict(sample_image, synthetic_feature_ae_checkpoint)

        assert prediction.latency_ms > 0.0
        assert isinstance(prediction.latency_ms, float)


def test_save_feature_ae_heatmap_overlay_writes_png(tmp_path: Path) -> None:
    image_path = tmp_path / "piece.jpg"
    heatmap_path = tmp_path / "heatmap.png"
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(image_path)

    save_feature_ae_heatmap_overlay(image_path, torch.rand(8, 8), heatmap_path)

    assert heatmap_path.exists()
    with Image.open(heatmap_path) as image:
        assert image.size == (16, 16)


def test_save_feature_ae_heatmap_overlay_uses_roi_percentile_display(tmp_path: Path) -> None:
    image_path = tmp_path / "piece.jpg"
    heatmap_path = tmp_path / "heatmap.png"
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(image_path)

    score_map = torch.full((8, 8), 0.10)
    score_map[4, 4] = 10.0
    save_feature_ae_heatmap_overlay(
        image_path,
        score_map,
        heatmap_path,
        threshold_orange=100.0,
        threshold_red=200.0,
    )

    with Image.open(heatmap_path) as image:
        rgb = image.convert("RGB")
        hot_pixel = rgb.getpixel((8, 8))
        background_pixel = rgb.getpixel((1, 1))
    assert hot_pixel[0] > 170
    assert hot_pixel[1] > 120
    assert hot_pixel[2] < 80
    assert background_pixel == (120, 120, 120)


def test_normalize_feature_ae_display_map_applies_roi_and_display_threshold() -> None:
    score_map = torch.tensor(
        [
            [0.1, 0.1, 9.0],
            [0.1, 8.0, 0.1],
            [0.1, 0.1, 0.1],
        ]
    ).numpy()
    roi_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    ).numpy()

    display = normalize_feature_ae_display_map(
        score_map,
        roi_mask=roi_mask,
        low_percentile=85.0,
        high_percentile=99.8,
        gamma=1.4,
        display_threshold=0.60,
    )

    assert display[0, 2] == 0.0
    assert display[1, 1] > 0.60
    assert display[0, 0] == 0.0


def test_champion_feature_map_fusion_uses_layer_weights() -> None:
    fused = fuse_numpy_layer_maps(
        {
            "layer2": torch.ones(2, 2).numpy(),
            "layer3": (torch.ones(2, 2) * 10).numpy(),
        },
        layer_weights={"layer2": 0.65, "layer3": 0.35},
    )

    assert fused[0, 0] == pytest.approx(4.15)


def test_champion_roi_soft_map_weights_scores() -> None:
    score_map = torch.tensor([[10.0, 10.0], [10.0, 10.0]]).numpy()
    roi_probability = torch.tensor([[1.0, 0.5], [0.0, 0.25]]).numpy()

    weighted = apply_champion_roi(score_map, roi_probability=roi_probability, roi_mode="soft_map")
    score = score_numpy_map_topk(
        weighted,
        roi_probability=roi_probability,
        score_image="topk_mean",
        topk_fraction=1.0,
    )

    assert weighted.tolist() == [[10.0, 5.0], [0.0, 2.5]]
    assert score == pytest.approx((10.0 + 5.0 + 2.5) / 3.0)


def test_roi_and_feature_ae_runtime_fall_back_to_cpu_when_cuda_is_unavailable() -> None:
    """CPU-built task images must not deserialize CUDA checkpoints onto CUDA."""
    segmentation_source = (Path(__file__).parents[2] / "src" / "iqa" / "inference" / "segmentation.py").read_text(
        encoding="utf-8"
    )
    feature_ae_source = (Path(__file__).parents[2] / "src" / "iqa" / "inference" / "feature_ae.py").read_text(
        encoding="utf-8"
    )

    for source in (segmentation_source, feature_ae_source):
        assert 'torch_device.type == "cuda" and not torch.cuda.is_available()' in source
        assert 'torch.device("cpu")' in source
        assert 'map_location="cpu"' in source
