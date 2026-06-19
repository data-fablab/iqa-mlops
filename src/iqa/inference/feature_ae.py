"""Feature-AE image prediction for the reproducible IQA source path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from iqa.datasets import load_image_tensor
from iqa.inference.helpers import compute_status, measure_inference_time
from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    FEATURE_AE_MODEL_TYPE,
    ResNetTeacherFeatures,
    feature_anomaly_map,
    load_rd_feature_ae_gated,
    normalize_feature_layers,
)
from iqa.training.feature_ae_contracts import CANONICAL_FEATURE_AE_PREPROCESSING


@dataclass(frozen=True)
class FeatureAEPrediction:
    image_path: str
    model_type: str
    score: float
    status: str
    threshold_orange: float
    threshold_red: float
    latency_ms: float
    roi_status: str | None = None
    heatmap_uri: str | None = None
    threshold_source: str = "explicit_or_default"

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def predict_feature_ae_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    image_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.image_size,
    context_size: int = CANONICAL_FEATURE_AE_PREPROCESSING.context_size,
    preprocessing_mode: str = CANONICAL_FEATURE_AE_PREPROCESSING.preprocessing_mode,
    threshold_orange: float = 0.02,
    threshold_red: float = 0.05,
    threshold_source: str = "explicit_or_default",
    roi_mask_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
    score_image: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
    topk_fraction: float = CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
    heatmap_output_path: str | Path | None = None,
    heatmap_uri: str | None = None,
    device: str = "cpu",
    pretrained_teacher: bool = False,
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS,
) -> FeatureAEPrediction:
    layers = normalize_feature_layers(layers)
    torch_device = torch.device(device)
    image = load_image_tensor(image_path, image_size=image_size).unsqueeze(0).to(torch_device)
    context_image_size = context_size if preprocessing_mode == "tiled_context" else image_size
    context_image = load_image_tensor(image_path, image_size=context_image_size).unsqueeze(0).to(torch_device)

    model = load_rd_feature_ae_gated(checkpoint_path, layers=layers, map_location=torch_device).to(torch_device)
    teacher = ResNetTeacherFeatures(layers=layers, pretrained=pretrained_teacher).to(torch_device)
    model.eval()
    teacher.eval()

    with torch.no_grad():
        with measure_inference_time() as timing:
            teacher_features = teacher(image)
            reconstructed = model(image, context_images=context_image)
            anomaly_map = feature_anomaly_map(teacher_features, reconstructed)

    score_map = anomaly_map.squeeze().detach().cpu()
    visual_score_map = prepare_feature_ae_score_map(
        score_map,
        roi_mask_path=roi_mask_path,
        score_smoothing=score_smoothing,
    )
    if heatmap_output_path is not None:
        save_feature_ae_heatmap_overlay(
            image_path,
            visual_score_map,
            heatmap_output_path,
            roi_mask_path=roi_mask_path,
            threshold_orange=threshold_orange,
            threshold_red=threshold_red,
        )
    score = score_feature_ae_map(
        score_map,
        roi_mask_path=roi_mask_path,
        score_smoothing=score_smoothing,
        score_image=score_image,
        topk_fraction=topk_fraction,
    )
    status = compute_status(score, threshold_orange=threshold_orange, threshold_red=threshold_red)
    return FeatureAEPrediction(
        image_path=str(image_path),
        model_type=FEATURE_AE_MODEL_TYPE,
        score=score,
        status=status,
        threshold_orange=float(threshold_orange),
        threshold_red=float(threshold_red),
        latency_ms=timing["elapsed_ms"],
        roi_status="roi_scored" if roi_mask_path is not None else None,
        heatmap_uri=heatmap_uri or (str(heatmap_output_path) if heatmap_output_path is not None else None),
        threshold_source=threshold_source,
    )


def score_feature_ae_map(
    score_map: torch.Tensor,
    *,
    roi_mask_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
    score_image: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
    topk_fraction: float = CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
) -> float:
    score_map = prepare_feature_ae_score_map(
        score_map,
        roi_mask_path=roi_mask_path,
        score_smoothing=score_smoothing,
    )
    valid_mask = load_roi_mask(roi_mask_path, target_shape=score_map.shape) if roi_mask_path is not None else None
    return score_image_map(score_map, score_image=score_image, topk_fraction=topk_fraction, valid_mask=valid_mask)


def prepare_feature_ae_score_map(
    score_map: torch.Tensor,
    *,
    roi_mask_path: str | Path | None = None,
    score_smoothing: str = CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
) -> torch.Tensor:
    score_map = score_map.to(dtype=torch.float32)
    if score_map.ndim != 2:
        raise ValueError(f"Feature-AE score map must be 2D, got shape={tuple(score_map.shape)}")
    score_map = smooth_score_map(score_map, score_smoothing)
    valid_mask = load_roi_mask(roi_mask_path, target_shape=score_map.shape) if roi_mask_path is not None else None
    if valid_mask is not None:
        score_map = score_map.masked_fill(~valid_mask, 0.0)
    return score_map


def smooth_score_map(score_map: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return score_map
    if mode == "median3":
        padded = F.pad(score_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
        windows = F.unfold(padded, kernel_size=3).squeeze(0)
        return windows.median(dim=0).values.reshape_as(score_map)
    raise ValueError(f"Unsupported Feature-AE score_smoothing: {mode!r}")


def score_image_map(
    score_map: torch.Tensor,
    *,
    score_image: str,
    topk_fraction: float,
    valid_mask: torch.Tensor | None = None,
) -> float:
    values = score_map[valid_mask] if valid_mask is not None else score_map.flatten()
    if values.numel() == 0:
        values = score_map.flatten()
    if score_image == "max":
        return float(values.max().item())
    if score_image == "mean":
        return float(values.mean().item())
    if score_image != "topk_mean":
        raise ValueError(f"Unsupported Feature-AE score_image: {score_image!r}")
    k = max(1, int(np.ceil(values.numel() * float(topk_fraction))))
    return float(torch.topk(values, min(k, values.numel())).values.mean().item())


def load_roi_mask(roi_mask_path: str | Path, *, target_shape: torch.Size | tuple[int, int]) -> torch.Tensor:
    mask = Image.open(roi_mask_path).convert("L")
    height, width = int(target_shape[0]), int(target_shape[1])
    if mask.size != (width, height):
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
    array = np.asarray(mask)
    return torch.from_numpy(array > 0)


def save_feature_ae_heatmap_overlay(
    image_path: str | Path,
    score_map: torch.Tensor,
    output_path: str | Path,
    *,
    roi_mask_path: str | Path | None = None,
    threshold_orange: float | None = None,
    threshold_red: float | None = None,
) -> None:
    """Save a red anomaly overlay PNG for operator review."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    score_array = score_map.detach().cpu().numpy().astype(np.float32)
    if threshold_red is not None and float(threshold_red) > 0.0:
        normalized = np.sqrt(np.clip(score_array / max(float(threshold_red), 1e-6), 0.0, 1.0))
    else:
        positive = score_array[score_array > 0]
        if positive.size:
            low = float(np.percentile(positive, 50))
            high = float(np.percentile(positive, 99))
            if high <= low:
                high = float(positive.max() or 1.0)
            normalized = np.clip((score_array - low) / max(high - low, 1e-6), 0.0, 1.0)
        else:
            normalized = np.zeros_like(score_array, dtype=np.float32)

    base = Image.open(image_path).convert("RGB")
    if base.size != (score_array.shape[1], score_array.shape[0]):
        alpha_image = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L")
        alpha_image = alpha_image.resize(base.size, Image.Resampling.BILINEAR)
        normalized = np.asarray(alpha_image, dtype=np.float32) / 255.0
    if roi_mask_path is not None:
        roi_mask = Image.open(roi_mask_path).convert("L").resize(base.size, Image.Resampling.NEAREST)
        normalized = normalized * (np.asarray(roi_mask, dtype=np.float32) > 0)
    base_array = np.asarray(base, dtype=np.float32)
    red = np.zeros_like(base_array)
    red[..., 0] = 255.0
    alpha = (normalized[..., None] * 0.55).astype(np.float32)
    overlay = (base_array * (1.0 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output)


__all__ = [
    "FeatureAEPrediction",
    "load_roi_mask",
    "prepare_feature_ae_score_map",
    "predict_feature_ae_image",
    "save_feature_ae_heatmap_overlay",
    "score_feature_ae_map",
    "score_image_map",
    "smooth_score_map",
]
