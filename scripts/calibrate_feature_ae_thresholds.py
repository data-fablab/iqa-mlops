"""Calibrate Feature-AE runtime decision thresholds from the calibration manifest."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqa.inference.feature_ae import predict_feature_ae_image
from iqa.inference.segmentation import predict_roi_image
from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
    model_manifest_path,
    resolve_feature_ae_checkpoint,
    resolve_roi_segmenter_checkpoint,
)
from iqa.training.feature_ae_contracts import CANONICAL_FEATURE_AE_PREPROCESSING

DEFAULT_CALIBRATION_MANIFEST = Path("data/metadata/calibration_set_v001.csv")
DEFAULT_OUTPUT_ROOT = Path(".cache/iqa/calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", default=DEFAULT_FEATURE_AE_MODEL_VERSION)
    parser.add_argument("--roi-model-version", default=DEFAULT_ROI_MODEL_VERSION)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, default=DEFAULT_CALIBRATION_MANIFEST)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--orange-quantile", type=float, default=0.95)
    parser.add_argument("--red-quantile", type=float, default=0.99)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = calibrate_feature_ae_thresholds(args)
    print(json.dumps(result, indent=2, sort_keys=True))


def calibrate_feature_ae_thresholds(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.orange_quantile < args.red_quantile < 1.0:
        raise ValueError("--orange-quantile and --red-quantile must satisfy 0 < orange < red < 1")
    rows = load_calibration_rows(args.calibration_manifest)
    if not rows:
        raise ValueError(f"Calibration manifest is empty: {args.calibration_manifest}")

    roi_checkpoint = resolve_roi_segmenter_checkpoint(args.roi_model_version, strict_checksum=True)
    feature_checkpoint = resolve_feature_ae_checkpoint(args.model_version, strict_checksum=True)
    output_dir = args.output_root / args.model_version
    roi_dir = output_dir / "roi_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    scored_images: list[dict[str, Any]] = []
    for row in rows:
        for relative_path in split_manifest_paths(row.get("relative_paths") or row.get("relative_path") or ""):
            if args.max_images is not None and len(scores) >= args.max_images:
                break
            image_path = args.image_root / relative_path
            if not image_path.exists():
                raise FileNotFoundError(f"Calibration image is missing: {image_path}")
            image_key = relative_path.replace("\\", "/").replace("/", "__")
            roi_mask = roi_dir / f"{row.get('piece_event_id') or row.get('event_id')}_{image_key}.png"
            roi = predict_roi_image(image_path, roi_checkpoint, device=args.device, output_mask=roi_mask)
            prediction = predict_feature_ae_image(
                image_path,
                feature_checkpoint,
                device=args.device,
                roi_mask_path=roi_mask,
                threshold_orange=float("inf"),
                threshold_red=float("inf"),
                threshold_source="calibration_pending",
            )
            scores.append(prediction.score)
            scored_images.append(
                {
                    "piece_event_id": row.get("piece_event_id") or row.get("event_id") or "",
                    "relative_path": relative_path,
                    "score": prediction.score,
                    "roi_ratio": roi.roi_ratio,
                    "roi_quality_status": roi.roi_quality_status,
                }
            )
        if args.max_images is not None and len(scores) >= args.max_images:
            break

    if not scores:
        raise ValueError("No conforming calibration images could be scored")
    thresholds = build_decision_thresholds(
        scores,
        model_version=args.model_version,
        calibration_set_id=args.calibration_manifest.stem,
        orange_quantile=args.orange_quantile,
        red_quantile=args.red_quantile,
    )
    report = {
        "model_version": args.model_version,
        "roi_model_version": args.roi_model_version,
        "calibration_manifest": str(args.calibration_manifest),
        "image_root": str(args.image_root),
        "decision_thresholds": thresholds,
        "scored_images": scored_images,
    }
    report_path = output_dir / "threshold_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_manifest:
        update_manifest_thresholds(args.model_version, thresholds)
    return {
        "model_version": args.model_version,
        "calibration_manifest": str(args.calibration_manifest),
        "report": str(report_path),
        "decision_thresholds": thresholds,
        "manifest_updated": bool(args.write_manifest),
    }


def load_calibration_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [row for row in rows if (row.get("source_class") or "").strip()]


def split_manifest_paths(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", "|").replace(",", "|").split("|") if part.strip()]


def build_decision_thresholds(
    scores: list[float],
    *,
    model_version: str,
    calibration_set_id: str,
    orange_quantile: float,
    red_quantile: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot calibrate thresholds without scores")
    return {
        "method": "calibration_good_quantiles",
        "model_version": model_version,
        "calibration_set_id": calibration_set_id,
        "score_contract_version": CANONICAL_FEATURE_AE_PREPROCESSING.version,
        "score_region": CANONICAL_FEATURE_AE_PREPROCESSING.score_region,
        "score_smoothing": CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
        "score_image": CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
        "topk_fraction": CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
        "orange_quantile": float(orange_quantile),
        "red_quantile": float(red_quantile),
        "threshold_orange": float(np.quantile(values, orange_quantile)),
        "threshold_red": float(np.quantile(values, red_quantile)),
        "sample_count": int(values.size),
        "score_min": float(values.min()),
        "score_median": float(np.median(values)),
        "score_max": float(values.max()),
        "created_at": datetime.now(UTC).isoformat(),
    }


def update_manifest_thresholds(model_version: str, thresholds: dict[str, Any]) -> Path:
    path = model_manifest_path(model_version)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["decision_thresholds"] = thresholds
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
