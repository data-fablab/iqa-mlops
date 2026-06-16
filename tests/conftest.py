from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

# NOTE: heavy imports (torch, PIL, iqa.models.feature_ae, iqa.datasets) are
# deferred into the fixtures that need them so test collection stays lightweight.


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


@pytest.fixture
def synthetic_feature_ae_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal Feature AE checkpoint for testing."""
    import torch
    from iqa.models.feature_ae import DEFAULT_FEATURE_LAYERS, ReverseDistillationGatedDualContextResNet18

    checkpoint_path = tmp_path / "checkpoint.pt"
    model = ReverseDistillationGatedDualContextResNet18(layers=DEFAULT_FEATURE_LAYERS)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


@pytest.fixture
def feature_ae_gates_config() -> dict:
    """Canonical promotion gates config for the feature_ae base model.

    Returns a fresh dict each call so tests can mutate it without leaking state.
    """
    return {
        "feature_ae": {
            "recall_defect_min": 1.0,
            "image_ap_max_regression": 0.02,
            "orange_rate_max": 0.10,
            "latency_p95_ms_max": 1000,
        }
    }


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Write a small RGB image to disk for single-image inference tests."""
    from PIL import Image

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (32, 32), color=(128, 96, 64)).save(image_path)
    return image_path


@pytest.fixture
def synthetic_validation_manifest(tmp_path: Path) -> Path:
    """Create a minimal validation_set_v001 manifest CSV."""
    manifest_path = tmp_path / "validation_manifest.csv"
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
        for i in range(4):
            writer.writerow({
                "image_id": f"val_img_{i:03d}",
                "relative_path": f"image_{i:03d}.png",
                "event_id": f"val_evt_{i:03d}",
                "label": "good" if i % 2 == 0 else "defect",
                "split_set": "validation_set_v001",
                "source_class": "casting",
                "is_defective": i % 2 == 1,
                "scenario_id": "test",
                "dataset_version": "test_v001",
            })
    return manifest_path


@pytest.fixture
def synthetic_image_root(tmp_path: Path, synthetic_validation_manifest: Path) -> Path:
    """Create synthetic validation images."""
    from PIL import Image
    from iqa.datasets import FEATURE_AE_TILE_SIZE

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(4):
        img_path = image_dir / f"image_{i:03d}.png"
        img = Image.new("RGB", (FEATURE_AE_TILE_SIZE, FEATURE_AE_TILE_SIZE), color="green")
        img.save(img_path)
    return image_dir


@pytest.fixture
def mlflow_tracking_uri(tmp_path: Path) -> str:
    """Provide MLflow tracking URI (PostgreSQL if configured, else SQLite for tests).

    Configure with MLFLOW_TRACKING_URI env var for PostgreSQL:
      export MLFLOW_TRACKING_URI='postgresql://user:password@localhost:5432/mlflow'

    Local tests use SQLite for speed.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    # Fallback to SQLite for local tests
    return f"sqlite:///{tmp_path}/mlflow.db"
