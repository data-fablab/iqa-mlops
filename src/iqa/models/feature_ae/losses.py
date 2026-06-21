"""Feature-AE runtime losses and anomaly maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_anomaly_map(
    teacher_features: dict[str, torch.Tensor],
    reconstructed_features: dict[str, torch.Tensor],
    *,
    layer_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    maps = []
    target_size = None
    weights = layer_weights or {}
    total_weight = 0.0
    for layer, teacher in teacher_features.items():
        reconstructed = reconstructed_features[layer]
        layer_error = (teacher - reconstructed).pow(2).mean(dim=1, keepdim=True)
        target_size = target_size or layer_error.shape[-2:]
        weight = float(weights.get(layer, 1.0))
        total_weight += weight
        maps.append(weight * F.interpolate(layer_error, size=target_size, mode="bilinear", align_corners=False))
    return torch.stack(maps, dim=0).sum(dim=0) / max(total_weight, 1e-12)


def feature_reconstruction_loss(
    teacher_features: dict[str, torch.Tensor],
    reconstructed_features: dict[str, torch.Tensor],
    *,
    cosine_weight: float = 0.0,
    pixel_weight: torch.Tensor | None = None,
    layer_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    losses = []
    weights = layer_weights or {}
    for layer, teacher in teacher_features.items():
        reconstructed = reconstructed_features[layer]
        layer_error = (reconstructed - teacher).pow(2).mean(dim=1, keepdim=True)
        if pixel_weight is not None:
            weight = pixel_weight.to(device=layer_error.device, dtype=layer_error.dtype)
            if weight.ndim == 3:
                weight = weight.unsqueeze(1)
            if weight.shape[-2:] != layer_error.shape[-2:]:
                weight = F.interpolate(weight, size=layer_error.shape[-2:], mode="nearest")
            l2 = (layer_error * weight).sum() / weight.sum().clamp_min(1.0)
        else:
            l2 = layer_error.mean()
        if cosine_weight:
            cosine = 1.0 - F.cosine_similarity(
                reconstructed.flatten(1),
                teacher.flatten(1),
                dim=1,
            ).mean()
            l2 = l2 + float(cosine_weight) * cosine
        losses.append(float(weights.get(layer, 1.0)) * l2)
    return torch.stack(losses).sum() / max(1.0, sum(float(weights.get(layer, 1.0)) for layer in teacher_features))


__all__ = ["feature_anomaly_map", "feature_reconstruction_loss"]
