"""Test suite for FeatureAECandidate interface (IQA2_KEN01)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

from iqa.datasets import FEATURE_AE_TILE_SIZE
from iqa.inference.feature_ae import FeatureAEPrediction
from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    ReverseDistillationGatedDualContextResNet18,
)
from iqa.models.feature_ae_candidate import FeatureAECandidate
from iqa.training.feature_ae_evaluation import FeatureAEEvaluationConfig


@pytest.fixture
def minimal_candidate() -> FeatureAECandidate:
    """Minimal FeatureAECandidate for testing."""
    model = ReverseDistillationGatedDualContextResNet18(layers=DEFAULT_FEATURE_LAYERS)
    return FeatureAECandidate(model=model)


@pytest.fixture
def test_image_path() -> Generator[Path, None, None]:
    """Create a temporary test image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "test_image.png"
        img = Image.new("RGB", (FEATURE_AE_TILE_SIZE, FEATURE_AE_TILE_SIZE), color="red")
        img.save(image_path)
        yield image_path


class TestFeatureAECandidateInterface:
    """Test FeatureAECandidate public interface."""

    def test_candidate_has_required_methods(self, minimal_candidate):
        """FeatureAECandidate has all required methods."""
        assert hasattr(minimal_candidate, "predict")
        assert hasattr(minimal_candidate, "eval")
        assert hasattr(minimal_candidate, "save")
        assert callable(minimal_candidate.predict)
        assert callable(minimal_candidate.eval)
        assert callable(minimal_candidate.save)

    def test_class_has_train_and_load_methods(self):
        """FeatureAECandidate class has train() and load() class methods."""
        assert hasattr(FeatureAECandidate, "train")
        assert hasattr(FeatureAECandidate, "load")
        assert callable(getattr(FeatureAECandidate, "train"))
        assert callable(getattr(FeatureAECandidate, "load"))


class TestFeatureAECandidateSaveLoad:
    """Test save/load round-trip functionality."""

    def test_save_creates_checkpoint(self, minimal_candidate):
        """save() creates checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

            result = minimal_candidate.save(checkpoint_path)

            assert checkpoint_path.exists()
            assert "checkpoint_path" in result
            assert str(checkpoint_path) in result["checkpoint_path"]

    def test_load_from_saved_checkpoint(self, minimal_candidate):
        """load() can load from saved checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            minimal_candidate.save(checkpoint_path)

            loaded = FeatureAECandidate.load(checkpoint_path)

            assert loaded is not None
            assert loaded.model is not None
            assert type(loaded.model).__name__ == "ReverseDistillationGatedDualContextResNet18"

    def test_roundtrip_preserves_model_type(self, minimal_candidate):
        """save/load round-trip preserves model architecture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            minimal_candidate.save(checkpoint_path)

            loaded = FeatureAECandidate.load(checkpoint_path)

            assert type(loaded.model) == type(minimal_candidate.model)


class TestFeatureAECandidatePredict:
    """Test prediction conformance to model contract."""

    def test_predict_returns_prediction_object(self, minimal_candidate, test_image_path):
        """predict() returns FeatureAEPrediction instance."""
        result = minimal_candidate.predict(test_image_path, device="cpu")
        assert isinstance(result, FeatureAEPrediction)

    def test_predict_has_required_fields(self, minimal_candidate, test_image_path):
        """predict() returns all required contract fields."""
        result = minimal_candidate.predict(test_image_path, device="cpu")

        required_fields = {
            "score",
            "status",
            "latency_ms",
            "roi_status",
            "heatmap_uri",
            "image_path",
            "model_type",
        }
        for field in required_fields:
            assert hasattr(result, field), f"Missing {field} field"

    def test_predict_score_is_float(self, minimal_candidate, test_image_path):
        """predict() score is a float."""
        result = minimal_candidate.predict(test_image_path, device="cpu")
        assert isinstance(result.score, float)

    def test_predict_status_is_valid(self, minimal_candidate, test_image_path):
        """predict() status is one of Vert, Orange, Rouge."""
        result = minimal_candidate.predict(test_image_path, device="cpu")
        assert result.status in {"Vert", "Orange", "Rouge"}

    def test_predict_latency_is_positive(self, minimal_candidate, test_image_path):
        """predict() latency_ms is positive."""
        result = minimal_candidate.predict(test_image_path, device="cpu")
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms > 0

    def test_predict_respects_threshold_parameters(self, minimal_candidate, test_image_path):
        """predict() respects custom threshold parameters."""
        # Low thresholds = more Vert predictions
        result_high = minimal_candidate.predict(
            test_image_path, device="cpu", threshold_orange=100.0, threshold_red=200.0
        )
        assert result_high.status == "Vert"

        # High thresholds = more Rouge predictions
        result_low = minimal_candidate.predict(
            test_image_path, device="cpu", threshold_orange=0.001, threshold_red=0.002
        )
        assert result_low.status == "Rouge"

    def test_predict_different_images_same_contract(self, minimal_candidate):
        """predict() maintains contract across different image sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with multiple sizes
            for size in [64, FEATURE_AE_TILE_SIZE]:
                image_path = Path(tmpdir) / f"test_{size}.png"
                img = Image.new("RGB", (size, size), color="blue")
                img.save(image_path)

                result = minimal_candidate.predict(image_path, image_size=size, device="cpu")

                assert isinstance(result, FeatureAEPrediction)
                assert result.status in {"Vert", "Orange", "Rouge"}


class TestFeatureAECandidateTrain:
    """Test training interface."""

    def test_train_is_classmethod(self):
        """train() is a classmethod."""
        import inspect

        assert isinstance(
            inspect.getattr_static(FeatureAECandidate, "train"), classmethod
        ), "train should be a classmethod"

    def test_train_callable(self):
        """train() is callable."""
        assert callable(FeatureAECandidate.train)


class TestFeatureAECandidateEval:
    """Test evaluation interface."""

    def test_eval_is_instance_method(self, minimal_candidate):
        """eval() is an instance method."""
        assert callable(minimal_candidate.eval)

    def test_eval_accepts_config(self, minimal_candidate):
        """eval() accepts FeatureAEEvaluationConfig."""
        import inspect

        sig = inspect.signature(minimal_candidate.eval)
        assert "config" in sig.parameters


__all__ = []
