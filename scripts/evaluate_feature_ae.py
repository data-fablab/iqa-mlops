"""Evaluate a Feature-AE checkpoint on an IQA validation/test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.training import FeatureAEEvaluationConfig, evaluate_feature_ae_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-predictions-dir", action="append", type=Path, default=[])
    parser.add_argument("--gt-masks-manifest", type=Path)
    parser.add_argument("--validation-set-id", default="validation_set_v001")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--context-size", type=int, default=768)
    parser.add_argument("--tile-stride", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pretrained-teacher", action="store_true")
    parser.add_argument("--calibrate-normal", action="store_true")
    parser.add_argument("--calibration-mode", default="per_layer")
    parser.add_argument("--calibration-stat", default="median_mad")
    parser.add_argument("--calibration-max-images", type=int, default=120)
    parser.add_argument("--score-region", default="functional_surface_prediction")
    parser.add_argument("--roi-threshold", type=float, default=0.30)
    parser.add_argument("--apply-score-region-to-map", action="store_true")
    parser.add_argument("--score-smoothing", default="median3")
    parser.add_argument("--score-image", default="topk_mean")
    parser.add_argument("--topk-fraction", type=float, default=0.005)
    parser.add_argument("--save-score-maps", action="store_true")
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--max-previews", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_feature_ae_checkpoint(
        FeatureAEEvaluationConfig(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            image_root=args.image_root,
            output_dir=args.output_dir,
            roi_predictions_dirs=tuple(args.roi_predictions_dir),
            gt_masks_manifest=args.gt_masks_manifest,
            validation_set_id=args.validation_set_id,
            image_size=args.image_size,
            context_size=args.context_size,
            tile_stride=args.tile_stride,
            batch_size=args.batch_size,
            device=args.device,
            layers=tuple(args.layers),
            pretrained_teacher=args.pretrained_teacher,
            calibrate_normal=args.calibrate_normal,
            calibration_mode=args.calibration_mode,
            calibration_stat=args.calibration_stat,
            calibration_max_images=args.calibration_max_images,
            score_region=args.score_region,
            roi_threshold=args.roi_threshold,
            apply_score_region_to_map=args.apply_score_region_to_map,
            score_smoothing=args.score_smoothing,
            score_image=args.score_image,
            topk_fraction=args.topk_fraction,
            save_score_maps=args.save_score_maps,
            save_previews=args.save_previews,
            max_previews=args.max_previews,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
