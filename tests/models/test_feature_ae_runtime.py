from __future__ import annotations

import pytest
import torch

from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    TEACHER_LAYER_CHANNELS,
    ReverseDistillationGatedDualContextResNet18,
    feature_anomaly_map,
    feature_reconstruction_loss,
    normalize_feature_layers,
)
from iqa.models.feature_ae.runtime import load_rd_feature_ae_gated


def test_feature_ae_constants() -> None:
    assert FEATURE_AE_MODEL_TYPE == "reverse_distill_resnet18_dual_context_gated"
    assert DEFAULT_FEATURE_LAYERS == ("layer2", "layer3")


def test_normalize_feature_layers_rejects_unknown_layer() -> None:
    with pytest.raises(ValueError, match="Unknown teacher feature layer"):
        normalize_feature_layers(["layer1", "unknown"])


def test_feature_ae_forward_and_losses() -> None:
    layers = ("layer2", "layer3")
    model = ReverseDistillationGatedDualContextResNet18(layers=layers)
    model.eval()
    images = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        reconstructed = model(images, context_images=images)
    teacher_features = {layer: torch.randn_like(reconstructed[layer]) for layer in layers}
    anomaly = feature_anomaly_map(teacher_features, reconstructed)
    loss = feature_reconstruction_loss(teacher_features, reconstructed)

    assert reconstructed["layer2"].shape[1] == TEACHER_LAYER_CHANNELS["layer2"]
    assert reconstructed["layer3"].shape[1] == TEACHER_LAYER_CHANNELS["layer3"]
    assert anomaly.shape == (1, 1, *reconstructed["layer2"].shape[-2:])
    assert loss.ndim == 0


def test_feature_ae_keeps_checkpoint_module_names() -> None:
    model = ReverseDistillationGatedDualContextResNet18()

    for module_name in ["encoder_stem", "decode_layer3", "context_film", "context_gate"]:
        assert hasattr(model, module_name)


def test_load_feature_ae_missing_checkpoint_error() -> None:
    with pytest.raises(FileNotFoundError, match="Feature-AE checkpoint not found"):
        load_rd_feature_ae_gated("missing-checkpoint.pt")
