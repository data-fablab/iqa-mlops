"""Calibrate Feature-AE with the reference tiled runtime contract."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
    model_manifest_path,
    resolve_feature_ae_checkpoint,
)
from iqa.models.feature_ae import REFERENCE_FEATURE_AE_CONTRACT
from iqa.training.feature_ae_evaluation import (
    FeatureAEEvaluationConfig,
    evaluate_feature_ae_checkpoint,
    parse_layer_loss_weights,
)

DEFAULT_VALIDATION_MANIFEST = Path("data/validation/validation_set_v001.csv")
DEFAULT_GT_MASKS_MANIFEST = Path("data/validation/validation_gt_masks_v001.csv")
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/calibration_reference")
BUSINESS_METRIC_PRIORITY = (
    "pixel_aupimo_1e-5_1e-3",
    "pixel_ap",
    "image_ap",
    "image_auroc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_FEATURE_AE_MODEL_VERSION)
    parser.add_argument("--roi-model-version", default=DEFAULT_ROI_MODEL_VERSION)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--gt-masks-manifest", type=Path, default=DEFAULT_GT_MASKS_MANIFEST)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--roi-mode", default=REFERENCE_FEATURE_AE_CONTRACT.roi_mode)
    parser.add_argument("--roi-threshold", type=float, default=REFERENCE_FEATURE_AE_CONTRACT.roi_threshold)
    parser.add_argument("--layer-weights", nargs="*", default=["layer2=0.65", "layer3=0.35"])
    parser.add_argument("--topk-fraction", type=float, default=REFERENCE_FEATURE_AE_CONTRACT.topk_fraction)
    parser.add_argument("--orange-quantile", type=float, default=0.95)
    parser.add_argument("--red-quantile", type=float, default=0.99)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = calibrate_feature_ae_reference(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


def calibrate_feature_ae_reference(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.orange_quantile < args.red_quantile < 1.0:
        raise ValueError("--orange-quantile and --red-quantile must satisfy 0 < orange < red < 1")
    assert_validation_has_defects(args.validation_manifest, args.gt_masks_manifest)

    checkpoint = resolve_feature_ae_checkpoint(args.model_version, strict_checksum=True)
    output_dir = args.output_root / args.model_version
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_weights = parse_layer_loss_weights(tuple(args.layer_weights))
    evaluation = evaluate_feature_ae_checkpoint(
        FeatureAEEvaluationConfig(
            checkpoint_path=checkpoint,
            manifest_path=args.validation_manifest,
            image_root=args.image_root,
            output_dir=output_dir / "evaluation",
            gt_masks_manifest=args.gt_masks_manifest,
            device=args.device,
            layer_weights=layer_weights,
            calibrate_normal=False,
            roi_threshold=args.roi_threshold,
            apply_score_region_to_map=False,
            score_smoothing=REFERENCE_FEATURE_AE_CONTRACT.score_smoothing,
            score_image=REFERENCE_FEATURE_AE_CONTRACT.score_image,
            topk_fraction=args.topk_fraction,
            save_score_maps=True,
            save_previews=True,
            max_previews=12,
        )
    )
    images = list(evaluation.get("images", []))
    if args.max_images is not None:
        images = images[: args.max_images]
    image_scores = [float(row["score"]) for row in images]
    normal_scores = [float(row["score"]) for row in images if not bool(row["is_defective"])]
    if not normal_scores:
        raise ValueError("Reference calibration requires at least one conforming validation image")

    metrics = dict(evaluation.get("metrics", {}))
    selected_metric, selected_metric_value = select_business_metric(metrics)
    thresholds = build_reference_thresholds(
        normal_scores,
        model_version=args.model_version,
        validation_manifest=args.validation_manifest,
        gt_masks_manifest=args.gt_masks_manifest,
        layer_weights=layer_weights,
        orange_quantile=args.orange_quantile,
        red_quantile=args.red_quantile,
        selected_metric=selected_metric,
        selected_metric_value=selected_metric_value,
    )
    write_calibration_matrix(output_dir / "calibration_matrix.csv", metrics, thresholds)
    predictions_path = materialize_predictions(output_dir / "predictions.npz", images, image_scores)
    summary = {
        "model_version": args.model_version,
        "roi_model_version": args.roi_model_version,
        "contract": REFERENCE_FEATURE_AE_CONTRACT.to_dict(),
        "validation_manifest": str(args.validation_manifest),
        "gt_masks_manifest": str(args.gt_masks_manifest),
        "metrics": metrics,
        "selected_metric": selected_metric,
        "selected_metric_value": selected_metric_value,
        "decision_thresholds": thresholds,
        "predictions": str(predictions_path),
        "created_at": datetime.now(UTC).isoformat(),
    }
    summary_path = output_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    threshold_report = output_dir / "threshold_calibration_report.json"
    threshold_report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_manifest:
        update_reference_manifest(args.model_version, thresholds)
    return {
        "model_version": args.model_version,
        "calibration_summary": str(summary_path),
        "calibration_matrix": str(output_dir / "calibration_matrix.csv"),
        "predictions": str(predictions_path),
        "selected_metric": selected_metric,
        "selected_metric_value": selected_metric_value,
        "manifest_updated": bool(args.write_manifest),
    }


def assert_validation_has_defects(validation_manifest: Path, gt_masks_manifest: Path) -> None:
    defective_rows = 0
    with validation_manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if str(row.get("is_defective") or row.get("label") or "").lower() in {"true", "defective"}:
                defective_rows += 1
    gt_rows = 0
    with gt_masks_manifest.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("gt_mask_path") or row.get("mask_path") or row.get("path"):
                gt_rows += 1
    if defective_rows <= 0 or gt_rows <= 0:
        raise ValueError("Reference calibration requires defective validation images with GT masks")


def select_business_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    for metric in BUSINESS_METRIC_PRIORITY:
        value = metrics.get(metric)
        if value is not None and np.isfinite(float(value)):
            return metric, float(value)
    raise ValueError("Reference calibration produced no business metric: pixel_aupimo, pixel_ap, image_ap or image_auroc")


def build_reference_thresholds(
    normal_scores: list[float],
    *,
    model_version: str,
    validation_manifest: Path,
    gt_masks_manifest: Path,
    layer_weights: dict[str, float],
    orange_quantile: float,
    red_quantile: float,
    selected_metric: str,
    selected_metric_value: float,
) -> dict[str, Any]:
    values = np.asarray(normal_scores, dtype=np.float64)
    return {
        "method": "reference_good_quantiles_with_business_metrics",
        "model_version": model_version,
        "calibration_set_id": validation_manifest.stem,
        "gt_masks_manifest": str(gt_masks_manifest),
        "score_contract_version": REFERENCE_FEATURE_AE_CONTRACT.version,
        "teacher_weights": REFERENCE_FEATURE_AE_CONTRACT.teacher_weights,
        "layers": list(REFERENCE_FEATURE_AE_CONTRACT.layers),
        "layer_weights": layer_weights,
        "roi_mode": REFERENCE_FEATURE_AE_CONTRACT.roi_mode,
        "roi_threshold": REFERENCE_FEATURE_AE_CONTRACT.roi_threshold,
        "score_smoothing": REFERENCE_FEATURE_AE_CONTRACT.score_smoothing,
        "score_image": REFERENCE_FEATURE_AE_CONTRACT.score_image,
        "topk_fraction": REFERENCE_FEATURE_AE_CONTRACT.topk_fraction,
        "orange_quantile": float(orange_quantile),
        "red_quantile": float(red_quantile),
        "threshold_orange": float(np.quantile(values, orange_quantile)),
        "threshold_red": float(np.quantile(values, red_quantile)),
        "selected_metric": selected_metric,
        "selected_metric_value": selected_metric_value,
        "sample_count": int(values.size),
        "score_min": float(values.min()),
        "score_median": float(np.median(values)),
        "score_max": float(values.max()),
        "created_at": datetime.now(UTC).isoformat(),
    }


def materialize_predictions(path: Path, images: list[dict[str, Any]], scores: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_scores=np.asarray(scores, dtype=np.float32),
        image_labels=np.asarray([bool(row["is_defective"]) for row in images], dtype=np.bool_),
        image_ids=np.asarray([str(row["image_id"]) for row in images], dtype=object),
        relative_paths=np.asarray([str(row["relative_path"]) for row in images], dtype=object),
    )
    return path


def write_calibration_matrix(path: Path, metrics: dict[str, Any], thresholds: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "score_contract_version",
        "selected_metric",
        "selected_metric_value",
        "pixel_aupimo_1e-5_1e-3",
        "pixel_ap",
        "image_ap",
        "image_auroc",
        "threshold_orange",
        "threshold_red",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "score_contract_version": thresholds["score_contract_version"],
                "selected_metric": thresholds["selected_metric"],
                "selected_metric_value": thresholds["selected_metric_value"],
                "pixel_aupimo_1e-5_1e-3": metrics.get("pixel_aupimo_1e-5_1e-3"),
                "pixel_ap": metrics.get("pixel_ap"),
                "image_ap": metrics.get("image_ap"),
                "image_auroc": metrics.get("image_auroc"),
                "threshold_orange": thresholds["threshold_orange"],
                "threshold_red": thresholds["threshold_red"],
            }
        )
    return path


def update_reference_manifest(model_version: str, thresholds: dict[str, Any]) -> Path:
    path = model_manifest_path(model_version)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["feature_ae_reference_contract"] = REFERENCE_FEATURE_AE_CONTRACT.to_dict()
    manifest["decision_thresholds"] = thresholds
    manifest["preprocessing_contract_version"] = REFERENCE_FEATURE_AE_CONTRACT.version
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()

