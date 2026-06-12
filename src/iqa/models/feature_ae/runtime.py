"""Feature-AE checkpoint loading."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from iqa.models.feature_ae.models import (
    DEFAULT_FEATURE_LAYERS,
    ReverseDistillationGatedDualContextResNet18,
)


def load_rd_feature_ae_gated(
    checkpoint_path: str | Path,
    *,
    layers: Iterable[str] = DEFAULT_FEATURE_LAYERS,
    map_location: str | torch.device = "cpu",
) -> ReverseDistillationGatedDualContextResNet18:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature-AE checkpoint not found: {path}")

    model = ReverseDistillationGatedDualContextResNet18(layers=layers)
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


__all__ = ["load_rd_feature_ae_gated"]
