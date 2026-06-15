from __future__ import annotations

import tomllib
from pathlib import Path

import torch
from PIL import Image

from iqa.datasets import (
    FEATURE_AE_CONTEXT_SIZE,
    FEATURE_AE_PREPROCESSING_MODES,
    FEATURE_AE_TILE_SIZE,
    CastingImageDataset,
)
from iqa.inference import predict_feature_ae_image
from iqa.models.feature_ae import ReverseDistillationGatedDualContextResNet18
from iqa.training.feature_ae import FeatureAETrainingConfig, train_feature_ae


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(128, 96, 64)).save(path)


def test_casting_dataset_reads_manifest_paths(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_rgb_image(image_root / "Casting_class1/train/good/sample.jpg")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "event_id,image_ids,relative_paths\n"
        "piece_event_001,iqa_casting_sample,Casting_class1/train/good/sample.jpg\n",
        encoding="utf-8",
    )

    dataset = CastingImageDataset(manifest, image_root, image_size=32)

    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["image"].shape == (3, 32, 32)
    assert sample["context_image"].shape == (3, 32, 32)
    assert sample["image_id"] == "iqa_casting_sample"


def test_feature_ae_preprocessing_names_are_size_agnostic() -> None:
    assert FEATURE_AE_TILE_SIZE == 384
    assert FEATURE_AE_CONTEXT_SIZE == 768
    assert "tiled_context" in FEATURE_AE_PREPROCESSING_MODES
    assert "tile_256_overlap" not in FEATURE_AE_PREPROCESSING_MODES


def test_feature_ae_training_writes_reproducible_checkpoint(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _write_rgb_image(image_root / "Casting_class1/train/good/sample.jpg")
    manifest = tmp_path / "manifest.csv"
    checkpoint = tmp_path / "rd_feature_ae.pt"
    manifest.write_text(
        "event_id,image_ids,relative_paths\n"
        "piece_event_001,iqa_casting_sample,Casting_class1/train/good/sample.jpg\n",
        encoding="utf-8",
    )

    result = train_feature_ae(
        FeatureAETrainingConfig(
            manifest_path=manifest,
            image_root=image_root,
            output_checkpoint=checkpoint,
            image_size=32,
            context_size=64,
            preprocessing_mode="tiled_context",
            batch_size=1,
            max_steps=1,
        )
    )

    assert checkpoint.exists()
    assert result["steps"] == 1
    assert result["train_samples"] == 1


def test_feature_ae_predict_image_returns_structured_prediction(tmp_path: Path) -> None:
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
        threshold_orange=0.0,
        threshold_red=1.0,
    )

    assert prediction.model_type == "reverse_distill_resnet18_dual_context_gated"
    assert prediction.status in {"green", "orange", "red"}
    assert prediction.score >= 0.0


def test_reproducible_ml_cli_commands_are_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["iqa-train-feature-ae"] == "scripts.train_feature_ae:main"
    assert scripts["iqa-predict-image"] == "scripts.predict_image:main"
    assert scripts["iqa-predict-roi"] == "scripts.predict_roi:main"
    assert scripts["iqa-generate-bootstrap-roi"] == "scripts.generate_bootstrap_roi:main"
    assert scripts["iqa-validate-ml-source"] == "scripts.validate_ml_source:main"
