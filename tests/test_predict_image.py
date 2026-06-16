"""Tests for predict_image functionality with score, roi_status, heatmap, and latency_ms."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from iqa.inference import FeatureAEPrediction, predict_feature_ae_image
from iqa.models.feature_ae import ReverseDistillationGatedDualContextResNet18


def _write_rgb_image(path: Path) -> None:
    """Write a test RGB image to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 96, 64)).save(path)


class TestPredictImage:
    """Test predict_image functionality respecting IQA1_KEN04 requirements."""

    def test_predict_image_returns_score(self, tmp_path: Path) -> None:
        """predict_image returns score field."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert hasattr(prediction, "score")
        assert isinstance(prediction.score, float)
        assert prediction.score >= 0.0

    def test_predict_image_returns_roi_status(self, tmp_path: Path) -> None:
        """predict_image returns roi_status field (can be None)."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert hasattr(prediction, "roi_status")
        assert prediction.roi_status is None or isinstance(prediction.roi_status, str)

    def test_predict_image_returns_heatmap_uri(self, tmp_path: Path) -> None:
        """predict_image returns heatmap_uri field (placeholder or None)."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert hasattr(prediction, "heatmap_uri")
        assert prediction.heatmap_uri is None or isinstance(prediction.heatmap_uri, str)

    def test_predict_image_returns_latency_ms(self, tmp_path: Path) -> None:
        """predict_image returns latency_ms field."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert hasattr(prediction, "latency_ms")
        assert isinstance(prediction.latency_ms, float)
        assert prediction.latency_ms > 0.0

    def test_predict_image_latency_measured_on_inference(self, tmp_path: Path) -> None:
        """latency_ms is measured during model inference."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert prediction.latency_ms > 0.0
        assert prediction.latency_ms < 10000.0

    def test_predict_image_respects_contract(self, tmp_path: Path) -> None:
        """predict_image output respects IQA1_KEN01 model contract."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert isinstance(prediction, FeatureAEPrediction)
        assert prediction.model_type == "reverse_distill_resnet18_dual_context_gated"
        assert prediction.status in {"green", "orange", "red"}
        assert prediction.score >= 0.0
        assert prediction.latency_ms > 0.0

    def test_predict_image_to_dict_includes_all_fields(self, tmp_path: Path) -> None:
        """predict_image to_dict includes score, roi_status, heatmap_uri, latency_ms."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        result_dict = prediction.to_dict()
        assert "score" in result_dict
        assert "roi_status" in result_dict
        assert "heatmap_uri" in result_dict
        assert "latency_ms" in result_dict

    def test_predict_image_latency_ms_is_positive(self, tmp_path: Path) -> None:
        """latency_ms is a positive number representing inference time."""
        image_path = tmp_path / "sample.jpg"
        checkpoint = tmp_path / "rd_feature_ae.pt"
        _write_rgb_image(image_path)
        model = ReverseDistillationGatedDualContextResNet18()
        torch.save({"state_dict": model.state_dict()}, checkpoint)

        prediction = predict_feature_ae_image(
            image_path,
            checkpoint,
            image_size=32,
            context_size=64,
        )

        assert prediction.latency_ms > 0.0
        assert isinstance(prediction.latency_ms, float)
