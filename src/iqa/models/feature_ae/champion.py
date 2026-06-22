"""Champion Feature-AE score-map contract shared by runtime and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from iqa.datasets.casting import (
    FEATURE_AE_CONTEXT_SIZE,
    FEATURE_AE_TILE_SIZE,
    centered_box,
    image_crop_to_tensor,
    tile_boxes,
)
from iqa.models.feature_ae.models import DEFAULT_FEATURE_LAYERS, normalize_feature_layers

FEATURE_AE_CHAMPION_LAYER_WEIGHTS = {"layer2": 0.65, "layer3": 0.35}
FEATURE_AE_CHAMPION_ROI_MODE = "soft_map"
FEATURE_AE_CHAMPION_TEACHER_WEIGHTS = "IMAGENET1K_V1"
FEATURE_AE_REFERENCE_SCORE_MODE = "sqrt_l2_plus_cosine"
FEATURE_AE_REFERENCE_LAYER_NORMALIZATION = "good_p99"


@dataclass(frozen=True)
class FeatureAEChampionContract:
    version: str = "feature_ae_champion_v001"
    teacher_weights: str = FEATURE_AE_CHAMPION_TEACHER_WEIGHTS
    tile_size: int = FEATURE_AE_TILE_SIZE
    context_size: int = FEATURE_AE_CONTEXT_SIZE
    tile_stride: int = FEATURE_AE_TILE_SIZE
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS
    layer_weights: dict[str, float] | None = None
    score_smoothing: str = "median3"
    roi_mode: str = FEATURE_AE_CHAMPION_ROI_MODE
    roi_threshold: float = 0.5
    score_image: str = "topk_mean"
    topk_fraction: float = 0.005
    layer_score_mode: str = FEATURE_AE_REFERENCE_SCORE_MODE
    layer_normalization: str = FEATURE_AE_REFERENCE_LAYER_NORMALIZATION
    layer_normalization_stats: dict[str, float] | None = None
    cosine_weight: float = 0.5

    def normalized_layer_weights(self) -> dict[str, float]:
        layers = normalize_feature_layers(self.layers)
        weights = self.layer_weights or FEATURE_AE_CHAMPION_LAYER_WEIGHTS
        values = {layer: float(weights[layer]) for layer in layers}
        total = sum(values.values())
        if total <= 0:
            raise ValueError("Feature-AE champion layer weights must sum to a positive value")
        return {layer: value / total for layer, value in values.items()}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_weights"] = self.normalized_layer_weights()
        return payload


CHAMPION_FEATURE_AE_CONTRACT = FeatureAEChampionContract(
    layer_weights=FEATURE_AE_CHAMPION_LAYER_WEIGHTS.copy()
)


def feature_layer_anomaly_maps(
    teacher_features: dict[str, torch.Tensor],
    reconstructed_features: dict[str, torch.Tensor],
    *,
    output_size: tuple[int, int] | None = None,
    layer_score_mode: str = FEATURE_AE_REFERENCE_SCORE_MODE,
    cosine_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return one anomaly map per teacher layer."""

    maps: dict[str, torch.Tensor] = {}
    for layer, teacher in teacher_features.items():
        reconstructed = reconstructed_features[layer]
        if layer_score_mode == "mean_squared_l2":
            layer_error = (teacher - reconstructed).pow(2).mean(dim=1, keepdim=True)
        elif layer_score_mode == FEATURE_AE_REFERENCE_SCORE_MODE:
            l2 = (teacher - reconstructed).pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
            cosine = 1.0 - F.cosine_similarity(reconstructed, teacher, dim=1, eps=1e-8).unsqueeze(1)
            layer_error = l2 + float(cosine_weight) * cosine
        else:
            raise ValueError(f"Unsupported Feature-AE layer score mode: {layer_score_mode!r}")
        size = output_size or layer_error.shape[-2:]
        maps[layer] = F.interpolate(layer_error, size=size, mode="bilinear", align_corners=False)
    return maps


def fuse_layer_anomaly_maps(
    layer_maps: dict[str, torch.Tensor],
    *,
    layer_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    layers = normalize_feature_layers(tuple(layer_maps))
    weights = layer_weights or FEATURE_AE_CHAMPION_LAYER_WEIGHTS
    total = sum(float(weights[layer]) for layer in layers)
    if total <= 0:
        raise ValueError("Feature-AE layer weights must sum to a positive value")
    fused = None
    for layer in layers:
        current = layer_maps[layer] * (float(weights[layer]) / total)
        fused = current if fused is None else fused + current
    if fused is None:
        raise ValueError("Cannot fuse empty Feature-AE layer maps")
    return fused


def reconstruct_tiled_feature_maps(
    *,
    image_path: str | Path,
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    device: torch.device,
    contract: FeatureAEChampionContract = CHAMPION_FEATURE_AE_CONTRACT,
) -> dict[str, np.ndarray]:
    """Run tiled 384/768 inference and reconstruct full-resolution layer maps."""

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    layer_names = normalize_feature_layers(contract.layers)
    sums = {layer: np.zeros((height, width), dtype=np.float32) for layer in layer_names}
    counts = np.zeros((height, width), dtype=np.float32)

    for box in tile_boxes(width, height, tile_size=contract.tile_size, stride=contract.tile_stride):
        context_box = centered_box(box, size=contract.context_size)
        tile = image_crop_to_tensor(image, box, output_size=contract.tile_size).unsqueeze(0).to(device)
        context = image_crop_to_tensor(image, context_box, output_size=contract.context_size).unsqueeze(0).to(device)
        with torch.no_grad():
            layer_maps = feature_layer_anomaly_maps(
                teacher(tile),
                model(tile, context_images=context),
                output_size=(contract.tile_size, contract.tile_size),
                layer_score_mode=contract.layer_score_mode,
                cosine_weight=contract.cosine_weight,
            )
        x0, y0, x1, y1 = box
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(width, x1), min(height, y1)
        px0, py0 = sx0 - x0, sy0 - y0
        px1, py1 = px0 + sx1 - sx0, py0 + sy1 - sy0
        for layer in layer_names:
            score = layer_maps[layer].squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            sums[layer][sy0:sy1, sx0:sx1] += score[py0:py1, px0:px1]
        counts[sy0:sy1, sx0:sx1] += 1.0

    return {layer: sums[layer] / np.maximum(counts, 1.0) for layer in layer_names}


def fuse_numpy_layer_maps(
    layer_maps: dict[str, np.ndarray],
    *,
    layer_weights: dict[str, float] | None = None,
    layer_normalization_stats: dict[str, float] | None = None,
) -> np.ndarray:
    layers = normalize_feature_layers(tuple(layer_maps))
    weights = layer_weights or FEATURE_AE_CHAMPION_LAYER_WEIGHTS
    total = sum(float(weights[layer]) for layer in layers)
    if total <= 0:
        raise ValueError("Feature-AE layer weights must sum to a positive value")
    fused = np.zeros_like(layer_maps[layers[0]], dtype=np.float32)
    for layer in layers:
        current = np.asarray(layer_maps[layer], dtype=np.float32)
        if layer_normalization_stats:
            scale = max(float(layer_normalization_stats.get(layer) or 1.0), 1e-8)
            current = current / scale
        fused += current * (float(weights[layer]) / total)
    return fused


def good_p99_layer_normalization_stats(
    layer_maps_by_image: dict[str, dict[str, np.ndarray]],
    *,
    normal_image_ids: set[str],
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
    percentile: float = 99.0,
) -> dict[str, float]:
    """Compute robust per-layer scales from conforming images only."""

    stats: dict[str, float] = {}
    for layer in normalize_feature_layers(layers):
        values = [
            np.asarray(layer_maps[layer], dtype=np.float32).reshape(-1)
            for image_id, layer_maps in layer_maps_by_image.items()
            if image_id in normal_image_ids and layer in layer_maps
        ]
        if not values:
            stats[layer] = 1.0
            continue
        scale = float(np.percentile(np.concatenate(values), float(percentile)))
        stats[layer] = max(scale, 1e-8)
    return stats


def smooth_numpy_score_map(score_map: np.ndarray, mode: str) -> np.ndarray:
    if mode in {"", "none"}:
        return score_map.astype(np.float32, copy=False)
    if mode != "median3":
        raise ValueError(f"Unsupported Feature-AE smoothing mode: {mode!r}")
    tensor = torch.from_numpy(score_map.astype(np.float32))[None, None]
    padded = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
    windows = F.unfold(padded, kernel_size=3).squeeze(0)
    return windows.median(dim=0).values.reshape(score_map.shape).numpy()


def load_roi_probability_map(
    path: str | Path | None,
    *,
    target_shape: tuple[int, int],
    threshold: float,
) -> np.ndarray | None:
    if path is None:
        return None
    roi = Image.open(path).convert("L")
    height, width = int(target_shape[0]), int(target_shape[1])
    if roi.size != (width, height):
        roi = roi.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(roi, dtype=np.float32) / 255.0
    return np.where(array >= float(threshold), array, 0.0).astype(np.float32)


def apply_champion_roi(
    score_map: np.ndarray,
    *,
    roi_probability: np.ndarray | None,
    roi_mode: str,
) -> np.ndarray:
    if roi_probability is None or roi_mode == "full":
        return score_map.astype(np.float32, copy=False)
    if roi_mode == "soft_map":
        return (score_map * roi_probability).astype(np.float32)
    if roi_mode == "hard_map":
        return np.where(roi_probability > 0, score_map, 0.0).astype(np.float32)
    if roi_mode == "hard_score_only":
        return score_map.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported Feature-AE ROI mode: {roi_mode!r}")


def score_numpy_map_topk(
    score_map: np.ndarray,
    *,
    roi_probability: np.ndarray | None,
    score_image: str,
    topk_fraction: float,
) -> float:
    valid = roi_probability > 0 if roi_probability is not None else None
    values = score_map[valid] if valid is not None else score_map.reshape(-1)
    if values.size == 0:
        values = score_map.reshape(-1)
    if values.size == 0:
        return 0.0
    if score_image == "max":
        return float(np.max(values))
    if score_image == "mean":
        return float(np.mean(values))
    if score_image != "topk_mean":
        raise ValueError(f"Unsupported Feature-AE score image mode: {score_image!r}")
    k = max(1, int(round(values.size * float(topk_fraction))))
    return float(np.partition(values, -k)[-k:].mean())


__all__ = [
    "CHAMPION_FEATURE_AE_CONTRACT",
    "FeatureAEChampionContract",
    "apply_champion_roi",
    "feature_layer_anomaly_maps",
    "fuse_layer_anomaly_maps",
    "fuse_numpy_layer_maps",
    "good_p99_layer_normalization_stats",
    "load_roi_probability_map",
    "reconstruct_tiled_feature_maps",
    "score_numpy_map_topk",
    "smooth_numpy_score_map",
]
