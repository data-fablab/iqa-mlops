"""Feature-AE metric evaluation and champion checkpoint selection."""

from __future__ import annotations

import json
import math
import shutil
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from iqa.datasets import FEATURE_AE_CONTEXT_SIZE, FEATURE_AE_TILE_SIZE, TiledFeatureAEDataset
from iqa.models.feature_ae import (
    DEFAULT_FEATURE_LAYERS,
    ResNetTeacherFeatures,
    feature_anomaly_map,
    load_rd_feature_ae_gated,
    normalize_feature_layers,
)
from iqa.roi import load_roi_mask_lookup


METRIC_BEST_FILES = {
    "image_auroc": "checkpoint_best_image_auroc.pt",
    "image_ap": "checkpoint_best_image_ap.pt",
    "pixel_ap": "checkpoint_best_pixel_ap.pt",
    "pixel_aupimo_1e-5_1e-3": "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
}


@dataclass(frozen=True)
class FeatureAEEvaluationConfig:
    checkpoint_path: Path
    manifest_path: Path
    image_root: Path
    output_dir: Path
    roi_predictions_dirs: tuple[Path, ...] = ()
    gt_masks_manifest: Path | None = None
    validation_set_id: str = "validation_set_v001"
    image_size: int = FEATURE_AE_TILE_SIZE
    context_size: int = FEATURE_AE_CONTEXT_SIZE
    tile_stride: int = FEATURE_AE_TILE_SIZE // 2
    batch_size: int = 8
    device: str = "cpu"
    layers: tuple[str, ...] = DEFAULT_FEATURE_LAYERS
    pretrained_teacher: bool = False
    calibrate_normal: bool = False
    calibration_mode: str = "per_layer"
    calibration_stat: str = "median_mad"
    calibration_max_images: int = 120
    score_region: str = "functional_surface_prediction"
    roi_threshold: float = 0.3
    apply_score_region_to_map: bool = False
    score_smoothing: str = "median3"
    score_image: str = "topk_mean"
    topk_fraction: float = 0.005
    threshold_orange: float = 0.02
    threshold_red: float = 0.05
    save_score_maps: bool = False
    save_previews: bool = False
    max_previews: int = 31


@dataclass(frozen=True)
class EvaluationReport:
    """Structured evaluation report for Feature AE model on validation set."""

    model_version: str
    average_precision: float
    recall: float
    orange_rate: float
    latency_ms: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_version": self.model_version,
            "average_precision": self.average_precision,
            "recall": self.recall,
            "orange_rate": self.orange_rate,
            "latency_ms": self.latency_ms,
            "sample_count": self.sample_count,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def parse_layer_loss_weights(values: list[str] | tuple[str, ...] | None) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"Layer weight must use layer=value syntax: {value!r}")
        layer, raw_weight = value.split("=", 1)
        layer = layer.strip()
        if layer not in normalize_feature_layers((layer,)):
            raise ValueError(f"Unknown layer in weight: {layer}")
        weights[layer] = float(raw_weight)
    return weights


def smooth_score_map(score_map: np.ndarray, mode: str = "median3") -> np.ndarray:
    if mode in {"none", ""}:
        return score_map.astype(np.float32, copy=False)
    if mode != "median3":
        raise ValueError(f"Unsupported score smoothing mode: {mode}")
    tensor = torch.from_numpy(score_map.astype(np.float32))[None, None]
    padded = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
    smoothed = padded.unfold(2, 3, 1).unfold(3, 3, 1).contiguous().view(1, 1, *score_map.shape, 9)
    return smoothed.median(dim=-1).values.squeeze().numpy()


def score_image_map(
    score_map: np.ndarray,
    *,
    mode: str = "topk_mean",
    topk_fraction: float = 0.005,
    roi_mask: np.ndarray | None = None,
) -> float:
    values = score_map[roi_mask > 0] if roi_mask is not None else score_map.reshape(-1)
    if values.size == 0:
        return 0.0
    if mode == "max":
        return float(np.max(values))
    if mode == "mean":
        return float(np.mean(values))
    if mode != "topk_mean":
        raise ValueError(f"Unsupported image scoring mode: {mode}")
    k = max(1, int(math.ceil(values.size * float(topk_fraction))))
    return float(np.partition(values, -k)[-k:].mean())


def median_mad_stats(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(mad * 1.4826, 1e-8)


def apply_median_mad(score_map: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    median, scale = stats
    return np.maximum((score_map - median) / scale, 0.0).astype(np.float32)


def compute_binary_metrics(
    image_labels: list[bool],
    image_scores: list[float],
    pixel_labels: list[np.ndarray],
    pixel_scores: list[np.ndarray],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "image_auroc": None,
        "image_ap": None,
        "pixel_ap": None,
        "pixel_aupimo_1e-5_1e-3": None,
    }
    y_true = np.asarray(image_labels, dtype=np.int32)
    y_score = np.asarray(image_scores, dtype=np.float32)
    if np.unique(y_true).size == 2:
        metrics["image_auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["image_ap"] = float(average_precision_score(y_true, y_score))

    if pixel_labels:
        p_true = np.concatenate([labels.reshape(-1) for labels in pixel_labels]).astype(np.int32)
        p_score = np.concatenate([scores.reshape(-1) for scores in pixel_scores]).astype(np.float32)
        if p_true.sum() > 0 and np.unique(p_true).size == 2:
            metrics["pixel_ap"] = float(average_precision_score(p_true, p_score))
            metrics["pixel_aupimo_1e-5_1e-3"] = _normalized_low_fpr_auc(p_true, p_score)
    return metrics


def compute_decision_metrics(
    image_labels: list[bool],
    image_scores: list[float],
    *,
    threshold_orange: float,
    threshold_red: float,
    latencies_ms: list[float] | None = None,
) -> dict[str, float | int]:
    """Compute decision metrics for the recall gate from per-image scores.

    Detection rule: a defect is detected when its score crosses threshold_orange
    (anything that is not "green"). A defective image that stays green is a false
    negative, which the ``recall == 1.0`` gate must catch.

    Args:
        image_labels: True where the image is defective.
        image_scores: Per-image anomaly scores aligned with ``image_labels``.
        threshold_orange: Score at/above which an image is flagged (Orange/Rouge).
        threshold_red: Score at/above which an image is Rouge.
        latencies_ms: Optional per-image inference latencies for the p95 latency.

    Returns:
        Dict with recall, false_negatives, orange_rate and latency_ms (p95).
    """
    labels = np.asarray(image_labels, dtype=bool)
    scores = np.asarray(image_scores, dtype=np.float64)

    detected = scores >= threshold_orange
    defect_count = int(labels.sum())
    false_negatives = int((labels & ~detected).sum())
    recall = 1.0 if defect_count == 0 else float((labels & detected).sum() / defect_count)

    orange = (scores >= threshold_orange) & (scores < threshold_red)
    orange_rate = float(orange.mean()) if scores.size else 0.0

    latency_ms = (
        float(np.percentile(np.asarray(latencies_ms, dtype=np.float64), 95))
        if latencies_ms
        else 0.0
    )

    return {
        "recall": recall,
        "false_negatives": false_negatives,
        "orange_rate": orange_rate,
        "latency_ms": latency_ms,
    }


def _normalized_low_fpr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    low, high = 1e-5, 1e-3
    fpr_grid = np.concatenate(([low], fpr[(fpr >= low) & (fpr <= high)], [high]))
    tpr_grid = np.interp(fpr_grid, fpr, tpr)
    return float(np.trapz(tpr_grid, fpr_grid) / (high - low))


def evaluate_feature_ae_checkpoint(config: FeatureAEEvaluationConfig) -> dict[str, Any]:
    layers = normalize_feature_layers(config.layers)
    device = torch.device(config.device)
    roi_lookup = load_roi_mask_lookup(tuple(config.roi_predictions_dirs))
    dataset = TiledFeatureAEDataset(
        config.manifest_path,
        config.image_root,
        tile_size=config.image_size,
        context_size=config.context_size,
        tile_stride=config.tile_stride,
        roi_masks=roi_lookup.masks,
        roi_status=roi_lookup.status,
        gt_masks=_load_gt_mask_lookup(config.gt_masks_manifest),
        roi_threshold=config.roi_threshold,
        train_only_normal=False,
    )
    model = load_rd_feature_ae_gated(config.checkpoint_path, layers=layers, map_location=device).to(device)
    teacher = ResNetTeacherFeatures(layers=layers, pretrained=config.pretrained_teacher).to(device)
    teacher.eval()

    aggregated: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for batch_start in range(0, len(dataset), config.batch_size):
            items = [dataset[index] for index in range(batch_start, min(len(dataset), batch_start + config.batch_size))]
            images = torch.stack([item["image"] for item in items]).to(device)
            contexts = torch.stack([item["context_image"] for item in items]).to(device)
            start = time.perf_counter()
            maps = feature_anomaly_map(teacher(images), model(images, context_images=contexts))
            maps = F.interpolate(maps, size=(config.image_size, config.image_size), mode="bilinear", align_corners=False)
            tile_latency_ms = (time.perf_counter() - start) * 1000.0 / max(len(items), 1)
            for item, score_tensor in zip(items, maps.cpu(), strict=True):
                image_id = str(item["image_id"])
                width, height = item["image_size"]
                x0, y0, x1, y1 = item["tile_box"]
                entry = aggregated.setdefault(
                    image_id,
                    {
                        "score_sum": np.zeros((height, width), dtype=np.float32),
                        "count": np.zeros((height, width), dtype=np.float32),
                        "roi": np.zeros((height, width), dtype=np.float32),
                        "gt": np.zeros((height, width), dtype=np.float32),
                        "is_defective": bool(item["is_defective"]),
                        "relative_path": str(item["relative_path"]),
                        "latency_ms": 0.0,
                    },
                )
                entry["latency_ms"] += tile_latency_ms
                score = score_tensor.squeeze(0).numpy()
                roi = item["roi_mask"].squeeze(0).numpy()
                gt = item["gt_mask"].squeeze(0).numpy()
                sx0, sy0 = max(0, x0), max(0, y0)
                sx1, sy1 = min(width, x1), min(height, y1)
                px0, py0 = sx0 - x0, sy0 - y0
                px1, py1 = px0 + sx1 - sx0, py0 + sy1 - sy0
                entry["score_sum"][sy0:sy1, sx0:sx1] += score[py0:py1, px0:px1]
                entry["count"][sy0:sy1, sx0:sx1] += 1.0
                entry["roi"][sy0:sy1, sx0:sx1] = np.maximum(entry["roi"][sy0:sy1, sx0:sx1], roi[py0:py1, px0:px1])
                entry["gt"][sy0:sy1, sx0:sx1] = np.maximum(entry["gt"][sy0:sy1, sx0:sx1], gt[py0:py1, px0:px1])

    raw_maps: dict[str, np.ndarray] = {}
    normal_pixels: list[np.ndarray] = []
    for image_id, entry in aggregated.items():
        score_map = entry["score_sum"] / np.maximum(entry["count"], 1.0)
        raw_maps[image_id] = score_map
        if not entry["is_defective"] and len(normal_pixels) < config.calibration_max_images:
            normal_pixels.append(score_map.reshape(-1))
    stats = median_mad_stats(np.concatenate(normal_pixels)) if config.calibrate_normal and normal_pixels else (0.0, 1.0)

    image_labels: list[bool] = []
    image_scores: list[float] = []
    image_latencies_ms: list[float] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []
    maps_dir = config.output_dir / "score_maps"
    previews_dir = config.output_dir / "previews"
    if config.save_score_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)
    if config.save_previews:
        previews_dir.mkdir(parents=True, exist_ok=True)

    for image_index, (image_id, entry) in enumerate(aggregated.items()):
        score_map = apply_median_mad(raw_maps[image_id], stats) if config.calibrate_normal else raw_maps[image_id]
        score_map = smooth_score_map(score_map, config.score_smoothing)
        roi = entry["roi"] > 0
        if config.apply_score_region_to_map and config.score_region == "functional_surface_prediction":
            score_map = np.where(roi, score_map, 0.0)
        image_score = score_image_map(
            score_map,
            mode=config.score_image,
            topk_fraction=config.topk_fraction,
            roi_mask=roi if config.score_region == "functional_surface_prediction" else None,
        )
        gt = (entry["gt"] > 0).astype(np.uint8)
        image_labels.append(bool(entry["is_defective"]))
        image_scores.append(image_score)
        image_latencies_ms.append(float(entry["latency_ms"]))
        pixel_labels.append(gt)
        pixel_scores.append(score_map.astype(np.float32))
        per_image.append(
            {
                "image_id": image_id,
                "relative_path": entry["relative_path"],
                "is_defective": bool(entry["is_defective"]),
                "score": image_score,
                "gt_positive_pixels": int(gt.sum()),
            }
        )
        if config.save_score_maps:
            np.save(maps_dir / f"{image_id}.npy", score_map.astype(np.float32))
        if config.save_previews and image_index < config.max_previews:
            _save_preview(previews_dir / f"{image_id}.png", score_map)

    metrics = compute_binary_metrics(image_labels, image_scores, pixel_labels, pixel_scores)
    decision = compute_decision_metrics(
        image_labels,
        image_scores,
        threshold_orange=config.threshold_orange,
        threshold_red=config.threshold_red,
        latencies_ms=image_latencies_ms,
    )
    metrics["image_recall"] = decision["recall"]
    metrics["false_negatives"] = decision["false_negatives"]
    metrics["orange_rate"] = decision["orange_rate"]
    metrics["latency_ms"] = decision["latency_ms"]
    result = {
        "checkpoint": str(config.checkpoint_path),
        "validation_set_id": config.validation_set_id,
        "calibration": {
            "enabled": config.calibrate_normal,
            "mode": config.calibration_mode,
            "stat": config.calibration_stat,
            "median": stats[0],
            "scale": stats[1],
        },
        "metrics": metrics,
        "images": per_image,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def update_metric_best_checkpoints(
    *,
    run_dir: Path,
    candidate_checkpoint: Path,
    metrics: dict[str, float | None],
    epoch: int,
) -> dict[str, Any]:
    best_path = run_dir / "metric_eval_best.json"
    best: dict[str, Any] = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else {}
    for metric, filename in METRIC_BEST_FILES.items():
        value = metrics.get(metric)
        if value is None or not math.isfinite(float(value)):
            continue
        previous = best.get(metric, {}).get("value")
        if previous is None or float(value) > float(previous):
            shutil.copy2(candidate_checkpoint, run_dir / filename)
            best[metric] = {"value": float(value), "epoch": int(epoch), "checkpoint": filename}
    if "image_ap" in best:
        shutil.copy2(run_dir / METRIC_BEST_FILES["image_ap"], run_dir / "checkpoint_best_image.pt")
        shutil.copy2(run_dir / METRIC_BEST_FILES["image_ap"], run_dir / "checkpoint.pt")
    elif "image_auroc" in best:
        shutil.copy2(run_dir / METRIC_BEST_FILES["image_auroc"], run_dir / "checkpoint_best_image.pt")
        shutil.copy2(run_dir / METRIC_BEST_FILES["image_auroc"], run_dir / "checkpoint.pt")
    if "pixel_aupimo_1e-5_1e-3" in best:
        shutil.copy2(run_dir / METRIC_BEST_FILES["pixel_aupimo_1e-5_1e-3"], run_dir / "checkpoint_best_localization.pt")
    elif "pixel_ap" in best:
        shutil.copy2(run_dir / METRIC_BEST_FILES["pixel_ap"], run_dir / "checkpoint_best_localization.pt")
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    return best


def _save_preview(path: Path, score_map: np.ndarray) -> None:
    normalized = score_map - float(score_map.min())
    scale = float(normalized.max())
    if scale > 0:
        normalized = normalized / scale
    Image.fromarray((normalized * 255.0).astype(np.uint8)).save(path)


def _load_gt_mask_lookup(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    masks: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            mask_value = row.get("gt_mask_path") or row.get("mask_path") or row.get("path") or ""
            if not mask_value:
                continue
            mask_path = Path(mask_value)
            if not mask_path.is_absolute():
                mask_path = path.parent / mask_path
            for key in (row.get("image_id") or "", row.get("relative_path") or ""):
                if key:
                    masks[key] = mask_path
    return masks


def evaluate_on_validation_set_v001(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    image_root: Path | str,
    output_dir: Path | str,
    model_version: str = "candidate",
) -> EvaluationReport:
    """Evaluate Feature AE model on frozen validation_set_v001.

    Args:
        checkpoint_path: Path to trained Feature AE checkpoint.
        manifest_path: Path to validation manifest CSV.
        image_root: Root directory for images.
        output_dir: Directory to save evaluation report.
        model_version: Version identifier for the model.

    Returns:
        EvaluationReport with AP, recall, Orange rate, latency metrics.

    Saves:
        JSON report at output_dir/evaluation_report.json.
    """
    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation using existing infrastructure
    config = FeatureAEEvaluationConfig(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        image_root=image_root,
        output_dir=output_dir,
        validation_set_id="validation_set_v001",
        device="cpu",
    )

    result = evaluate_feature_ae_checkpoint(config)
    metrics = result["metrics"]

    # Count evaluated images (falls back to manifest rows if none aggregated)
    num_samples = 0
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            num_samples = sum(1 for line in f) - 1  # -1 for header
    except Exception:
        pass

    # Extract key metrics from evaluation results. image_ap/auroc are None when
    # the validation set is single-class; fall back to 0.0 in that case.
    ap = metrics.get("image_ap") or 0.0
    recall = metrics.get("image_recall", 0.0)
    orange_rate = metrics.get("orange_rate", 0.0)
    latency_ms = metrics.get("latency_ms", 0.0)
    sample_count = max(len(result.get("images", [])), num_samples)

    # Create report
    report = EvaluationReport(
        model_version=model_version,
        average_precision=float(ap),
        recall=float(recall),
        orange_rate=float(orange_rate),
        latency_ms=float(latency_ms),
        sample_count=int(sample_count),
    )

    # Save report
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(report.to_json())

    return report


__all__ = [
    "EvaluationReport",
    "FeatureAEEvaluationConfig",
    "METRIC_BEST_FILES",
    "apply_median_mad",
    "compute_binary_metrics",
    "compute_decision_metrics",
    "evaluate_feature_ae_checkpoint",
    "evaluate_on_validation_set_v001",
    "median_mad_stats",
    "parse_layer_loss_weights",
    "score_image_map",
    "smooth_score_map",
    "update_metric_best_checkpoints",
]
