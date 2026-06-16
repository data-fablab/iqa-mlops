"""Tests for versioned Feature AE candidate training with traceability.

Scope: metadata/versioning only (git commit, training config, smoke training).
The public FeatureAECandidate contract (interface, save/load, predict, eval) is
covered in tests/test_feature_ae_candidate.py.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from iqa.datasets import FEATURE_AE_TILE_SIZE
from iqa.models.feature_ae import DEFAULT_FEATURE_LAYERS, ReverseDistillationGatedDualContextResNet18
from iqa.models.feature_ae_candidate import FeatureAECandidate
from iqa.training.feature_ae import FeatureAETrainingConfig


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_with_metadata(
    checkpoint_path: Path,
    config: FeatureAETrainingConfig,
) -> None:
    """Save checkpoint with metadata (helper for tests)."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    model = ReverseDistillationGatedDualContextResNet18(layers=DEFAULT_FEATURE_LAYERS)
    import torch
    torch.save(model.state_dict(), checkpoint_path)

    # Save metadata
    metadata = {
        "candidate_version": config.candidate_version,
        "dataset_version": config.dataset_version,
        "git_commit": _get_git_commit(),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }
    metadata_path = checkpoint_path.parent / f"{checkpoint_path.stem}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


class TestFeatureAECandidateVersionedMetadata:
    """Test versioned candidate metadata storage."""

    def test_save_candidate_with_metadata(self, tmp_path: Path) -> None:
        """Candidate checkpoint includes metadata with version info."""
        checkpoint = tmp_path / "candidate_v2.pt"
        metadata_path = tmp_path / "candidate_v2.metadata.json"

        config = FeatureAETrainingConfig(
            manifest_path=Path("dummy"),
            image_root=Path("dummy"),
            output_checkpoint=checkpoint,
            epochs=1,
            batch_size=2,
            dataset_version="dataset_v001",
            candidate_version="v2",
        )

        # Save with metadata
        _save_with_metadata(checkpoint, config)

        assert checkpoint.exists()
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata["candidate_version"] == "v2"
        assert metadata["dataset_version"] == "dataset_v001"

    def test_metadata_includes_git_commit(self, tmp_path: Path) -> None:
        """Candidate metadata includes git commit for reproducibility."""
        checkpoint = tmp_path / "candidate_v3.pt"
        metadata_path = tmp_path / "candidate_v3.metadata.json"

        config = FeatureAETrainingConfig(
            manifest_path=Path("dummy"),
            image_root=Path("dummy"),
            output_checkpoint=checkpoint,
            dataset_version="dataset_v001",
            candidate_version="v3",
        )

        _save_with_metadata(checkpoint, config)

        metadata = json.loads(metadata_path.read_text())
        assert "git_commit" in metadata
        # Git commit should be either 40 chars (SHA1) or "unknown"
        assert len(metadata["git_commit"]) in (40, 7)  # 40 for full SHA, 7+ for short

    def test_metadata_includes_training_config(self, tmp_path: Path) -> None:
        """Candidate metadata includes key training parameters."""
        checkpoint = tmp_path / "candidate_v2.pt"
        metadata_path = tmp_path / "candidate_v2.metadata.json"

        config = FeatureAETrainingConfig(
            manifest_path=Path("dummy"),
            image_root=Path("dummy"),
            output_checkpoint=checkpoint,
            epochs=5,
            batch_size=8,
            learning_rate=1e-4,
            dataset_version="dataset_v001",
            candidate_version="v2",
        )

        _save_with_metadata(checkpoint, config)

        metadata = json.loads(metadata_path.read_text())
        assert metadata["epochs"] == 5
        assert metadata["batch_size"] == 8
        assert metadata["learning_rate"] == 1e-4

    def test_metadata_filename_matches_checkpoint(self, tmp_path: Path) -> None:
        """Metadata file is named consistently with checkpoint."""
        checkpoint = tmp_path / "candidate_v2.pt"
        metadata_path = tmp_path / "candidate_v2.metadata.json"

        config = FeatureAETrainingConfig(
            manifest_path=Path("dummy"),
            image_root=Path("dummy"),
            output_checkpoint=checkpoint,
            dataset_version="dataset_v001",
            candidate_version="v2",
        )

        _save_with_metadata(checkpoint, config)

        # Metadata should be at checkpoint_stem.metadata.json
        assert metadata_path.exists()
        expected_metadata = checkpoint.parent / f"{checkpoint.stem}.metadata.json"
        assert metadata_path == expected_metadata


class TestFeatureAECandidateTrain:
    """Test FeatureAECandidate.train() with versioning."""

    @mock.patch("iqa.training.feature_ae.train_feature_ae")
    def test_train_saves_metadata_alongside_checkpoint(self, mock_train: mock.MagicMock, tmp_path: Path) -> None:
        """FeatureAECandidate.train() saves metadata alongside checkpoint."""
        checkpoint = tmp_path / "candidate_v2.pt"
        metadata_path = tmp_path / "candidate_v2.metadata.json"

        # Create a minimal checkpoint
        model = ReverseDistillationGatedDualContextResNet18(layers=DEFAULT_FEATURE_LAYERS)
        torch.save(model.state_dict(), checkpoint)

        # Mock train_feature_ae to return checkpoint path
        mock_train.return_value = {"checkpoint_path": str(checkpoint)}

        config = FeatureAETrainingConfig(
            manifest_path=Path("dummy"),
            image_root=Path("dummy"),
            output_checkpoint=checkpoint,
            dataset_version="dataset_v001",
            candidate_version="v2",
        )

        candidate = FeatureAECandidate.train(config)

        # Verify candidate was created
        assert candidate is not None

        # Verify metadata was saved
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["candidate_version"] == "v2"
        assert metadata["dataset_version"] == "dataset_v001"
        assert "git_commit" in metadata

    @mock.patch("iqa.training.feature_ae.train_feature_ae")
    def test_train_metadata_includes_all_config_params(self, mock_train: mock.MagicMock, tmp_path: Path) -> None:
        """Train metadata captures key configuration parameters."""
        checkpoint = tmp_path / "candidate_v3.pt"
        metadata_path = tmp_path / "candidate_v3.metadata.json"

        model = ReverseDistillationGatedDualContextResNet18(layers=DEFAULT_FEATURE_LAYERS)
        torch.save(model.state_dict(), checkpoint)
        mock_train.return_value = {"checkpoint_path": str(checkpoint)}

        config = FeatureAETrainingConfig(
            manifest_path=Path("dummy"),
            image_root=Path("dummy"),
            output_checkpoint=checkpoint,
            dataset_version="dataset_v001",
            candidate_version="v3",
            epochs=7,
            batch_size=16,
            learning_rate=2e-5,
            scenario_id="test_scenario",
        )

        FeatureAECandidate.train(config)

        metadata = json.loads(metadata_path.read_text())
        assert metadata["epochs"] == 7
        assert metadata["batch_size"] == 16
        assert metadata["learning_rate"] == 2e-5
        assert metadata["scenario_id"] == "test_scenario"


class TestFeatureAECandidateSmokeTraining:
    """Smoke tests for real training with minimal data."""

    def _create_minimal_dataset(self, tmp_path: Path, num_images: int = 4) -> tuple[Path, Path]:
        """Create minimal dataset with CSV manifest and images."""
        # Create images
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(num_images):
            img_path = image_dir / f"image_{i:03d}.png"
            img = Image.new("RGB", (FEATURE_AE_TILE_SIZE, FEATURE_AE_TILE_SIZE), color="blue")
            img.save(img_path)

        # Create manifest CSV
        manifest_path = tmp_path / "manifest.csv"
        fieldnames = [
            "image_id",
            "relative_path",
            "event_id",
            "label",
            "split_set",
            "source_class",
            "is_defective",
            "scenario_id",
            "dataset_version",
        ]
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(num_images):
                writer.writerow({
                    "image_id": f"img_{i:03d}",
                    "relative_path": f"image_{i:03d}.png",
                    "event_id": f"evt_{i:03d}",
                    "label": "good",
                    "split_set": "train",
                    "source_class": "casting",
                    "is_defective": False,
                    "scenario_id": "test",
                    "dataset_version": "test_v001",
                })

        return manifest_path, image_dir

    def test_smoke_train_short_epoch(self, tmp_path: Path) -> None:
        """Smoke test: train for 1 epoch with minimal data."""
        manifest, image_root = self._create_minimal_dataset(tmp_path)
        checkpoint = tmp_path / "candidate_v2.pt"
        metadata_path = tmp_path / "candidate_v2.metadata.json"

        config = FeatureAETrainingConfig(
            manifest_path=manifest,
            image_root=image_root,
            output_checkpoint=checkpoint,
            epochs=1,
            batch_size=2,
            max_steps=2,  # Very short training
            dataset_version="test_dataset_v001",
            candidate_version="v2",
            device="cpu",
            val_fraction=0.25,
        )

        # This should train without errors
        candidate = FeatureAECandidate.train(config)

        # Verify candidate created
        assert candidate is not None

        # Verify checkpoint exists
        assert checkpoint.exists()

        # Verify metadata exists and is correct
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert metadata["candidate_version"] == "v2"
        assert metadata["dataset_version"] == "test_dataset_v001"
        assert "git_commit" in metadata

    def test_smoke_train_produces_valid_model(self, tmp_path: Path) -> None:
        """Smoke test: trained model can load and predict."""
        manifest, image_root = self._create_minimal_dataset(tmp_path)
        checkpoint = tmp_path / "candidate_v3.pt"

        config = FeatureAETrainingConfig(
            manifest_path=manifest,
            image_root=image_root,
            output_checkpoint=checkpoint,
            epochs=1,
            batch_size=2,
            max_steps=2,
            dataset_version="test_dataset_v001",
            candidate_version="v3",
            device="cpu",
            val_fraction=0.25,
        )

        FeatureAECandidate.train(config)

        # Load the trained model
        loaded = FeatureAECandidate.load(checkpoint)
        assert loaded is not None
        assert loaded.model is not None

        # Smoke test prediction (should not crash)
        test_image = image_root / "image_000.png"
        prediction = loaded.predict(test_image)
        assert prediction is not None
        assert hasattr(prediction, "score")
        assert isinstance(prediction.score, float)
