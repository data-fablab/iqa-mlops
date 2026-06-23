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
    PREDICTION_SCHEMA_VERSION,
    evaluate_feature_ae_checkpoint,
    evaluate_feature_ae_predictions,
    parse_layer_loss_weights,
    score_image_map,
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
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--normal-tail-low-q", type=float, default=0.99)
    parser.add_argument("--normal-tail-high-q", type=float, default=0.999)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = calibrate_feature_ae_reference(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


def calibrate_feature_ae_reference(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.orange_quantile < args.red_quantile < 1.0:
        raise ValueError("--orange-quantile and --red-quantile must satisfy 0 < orange < red < 1")
    assert_validation_has_defects(args.validation_manifest, args.gt_masks_manifest)

    output_dir = args.output_root / args.model_version
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_weights = parse_layer_loss_weights(tuple(args.layer_weights))
    if args.predictions is not None:
        source_predictions = args.predictions
    else:
        checkpoint = resolve_feature_ae_checkpoint(args.model_version, strict_checksum=True)
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
        source_predictions = Path(str(evaluation["predictions_path"]))
    predictions_path = materialize_normal_tail_calibrated_predictions(
        source_predictions,
        output_dir / "predictions.npz",
        low_q=float(args.normal_tail_low_q),
        high_q=float(args.normal_tail_high_q),
        topk_fraction=float(args.topk_fraction),
    )
    evaluation = evaluate_feature_ae_predictions(
        predictions_path,
        threshold_orange=0.0,
        threshold_red=0.0,
    )
    images = list(evaluation.get("images", []))
    if args.max_images is not None:
        images = images[: args.max_images]
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
    summary = {
        "model_version": args.model_version,
        "roi_model_version": args.roi_model_version,
        "contract": REFERENCE_FEATURE_AE_CONTRACT.to_dict(),
        "posthoc_calibration": {
            "kind": "normal_tail_quantile",
            "low_q": float(args.normal_tail_low_q),
            "high_q": float(args.normal_tail_high_q),
            "source_npz": str(source_predictions),
            "score_contract_version": REFERENCE_FEATURE_AE_CONTRACT.version,
        },
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


def materialize_normal_tail_calibrated_predictions(
    source: Path,
    destination: Path,
    *,
    low_q: float,
    high_q: float,
    topk_fraction: float,
) -> Path:
    if not 0.0 < low_q < high_q < 1.0:
        raise ValueError("--normal-tail-low-q and --normal-tail-high-q must satisfy 0 < low < high < 1")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=True) as data:
        schema = str(data["schema_version"].item()) if "schema_version" in data.files and data["schema_version"].shape == () else ""
        if schema != PREDICTION_SCHEMA_VERSION or "score_maps" not in data.files or "masks" not in data.files:
            raise ValueError(
                "unsupported_prediction_schema: post-hoc calibration requires "
                f"schema_version={PREDICTION_SCHEMA_VERSION!r}, score_maps and masks"
            )
        raw_score_maps = np.asarray(data["score_maps"], dtype=np.float32)
        y_true = np.asarray(data["y_true"], dtype=bool)
        normal_pixels = raw_score_maps[~y_true].reshape(-1)
        if normal_pixels.size == 0:
            raise ValueError("normal_tail_quantile calibration requires at least one good image")
        low, high = np.quantile(normal_pixels, [low_q, high_q])
        scale = max(float(high - low), 1e-8)
        score_maps = ((raw_score_maps - float(low)) / scale).astype(np.float32)
        roi_masks = np.asarray(data["roi_masks"], dtype=np.float32) if "roi_masks" in data.files else np.ones_like(score_maps, dtype=np.float32)
        image_score = np.asarray(
            [
                score_image_map(score_map, topk_fraction=topk_fraction, roi_mask=roi > 0)
                for score_map, roi in zip(score_maps, roi_masks, strict=True)
            ],
            dtype=np.float32,
        )
        payload = {name: data[name] for name in data.files if name not in {"score_maps", "raw_score_maps", "image_score"}}
        payload["schema_version"] = np.asarray(PREDICTION_SCHEMA_VERSION, dtype=object)
        payload["raw_score_maps"] = raw_score_maps
        payload["score_maps"] = score_maps
        payload["image_score"] = image_score
        payload["calibration_kind"] = np.asarray("normal_tail_quantile", dtype=object)
        payload["calibration_low_q"] = np.asarray(float(low_q), dtype=np.float32)
        payload["calibration_high_q"] = np.asarray(float(high_q), dtype=np.float32)
        payload["calibration_low_value"] = np.asarray(float(low), dtype=np.float32)
        payload["calibration_high_value"] = np.asarray(float(high), dtype=np.float32)
    np.savez_compressed(destination, **payload)
    return destination


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
