"""Train the retained RD Feature-AE from IQA manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.training.feature_ae import FeatureAETrainingConfig, train_feature_ae
from iqa.training.feature_ae_evaluation import parse_layer_loss_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/feature_ae_bootstrap_events.csv"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, default=Path("models/feature_ae/checkpoint.pt"))
    parser.add_argument("--category", default="")
    parser.add_argument("--model-type", default="reverse_distill_resnet18_dual_context_gated")
    parser.add_argument("--teacher-backbone", default="resnet18")
    parser.add_argument("--image-size", "--input-size", "--tile-size", type=int, default=384)
    parser.add_argument("--context-size", "--context-tile-size", type=int, default=768)
    parser.add_argument("--preprocessing-mode", choices=["tiled_context"], default="tiled_context")
    parser.add_argument("--tile-stride", "--tile-train-stride", type=int, default=192)
    parser.add_argument("--tile-train-sampling", choices=["all"], default="all")
    parser.add_argument("--repeat-factor", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained-teacher", action="store_true")
    parser.add_argument("--loss", choices=["l2_cosine"], default="l2_cosine")
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--layer-loss-weights", nargs="*", default=["layer2=0.65", "layer3=0.35"])
    parser.add_argument("--augmentation-profile", default="none")
    parser.add_argument("--roi-predictions-dir", action="append", type=Path, default=[])
    parser.add_argument("--roi-threshold", type=float, default=0.30)
    parser.add_argument("--roi-loss-weight", type=float, default=1.0)
    parser.add_argument("--background-loss-weight", type=float, default=0.02)
    parser.add_argument("--min-roi-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler", choices=["plateau", "none"], default="plateau")
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--dataset-version", default="")
    parser.add_argument("--roi-model-version", default="")
    parser.add_argument("--feature-ae-version", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--metric-eval-manifest", type=Path)
    parser.add_argument("--metric-eval-category", default="")
    parser.add_argument("--metric-eval-device")
    parser.add_argument("--metric-eval-roi-predictions-dir", nargs="*", type=Path, default=[])
    parser.add_argument("--gt-masks-manifest", type=Path)
    parser.add_argument("--validation-set-id", default="validation_set_v001")
    parser.add_argument("--metric-eval-every-epochs", type=int, default=0)
    parser.add_argument("--metric-eval-start-epoch", type=int, default=1)
    parser.add_argument("--metric-eval-batch-size", type=int, default=8)
    parser.add_argument("--metric-eval-tile-stride", type=int)
    parser.add_argument("--metric-eval-layer-weights", nargs="*")
    parser.add_argument("--metric-eval-calibrate-normal", action="store_true")
    parser.add_argument("--metric-eval-calibration-mode", default="per_layer")
    parser.add_argument("--metric-eval-calibration-stat", default="median_mad")
    parser.add_argument("--metric-eval-calibration-max-images", type=int, default=120)
    parser.add_argument("--metric-eval-score-region", default="functional_surface_prediction")
    parser.add_argument("--metric-eval-apply-score-region-to-map", action="store_true")
    parser.add_argument("--metric-eval-score-smoothing", default="median3")
    parser.add_argument("--metric-eval-score-image", default="topk_mean")
    parser.add_argument("--metric-eval-topk-fraction", type=float, default=0.005)
    parser.add_argument("--metric-eval-save-score-maps", action="store_true")
    parser.add_argument("--metric-eval-save-previews", action="store_true")
    parser.add_argument("--metric-eval-max-previews", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_type != "reverse_distill_resnet18_dual_context_gated":
        raise ValueError(f"Unsupported Feature-AE model type: {args.model_type}")
    if args.teacher_backbone != "resnet18":
        raise ValueError(f"Unsupported teacher backbone: {args.teacher_backbone}")
    if args.augmentation_profile != "none":
        raise ValueError("Feature-AE champion reproduction only supports --augmentation-profile none.")
    result = train_feature_ae(
        FeatureAETrainingConfig(
            manifest_path=args.manifest,
            image_root=args.image_root,
            output_checkpoint=args.output_checkpoint,
            category=args.category,
            image_size=args.image_size,
            context_size=args.context_size,
            preprocessing_mode=args.preprocessing_mode,
            tile_stride=args.tile_stride,
            tile_train_sampling=args.tile_train_sampling,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            device=args.device,
            pretrained_teacher=args.pretrained_teacher,
            loss=args.loss,
            cosine_weight=args.cosine_weight,
            layer_loss_weights=parse_layer_loss_weights(args.layer_loss_weights),
            layers=tuple(args.layers),
            repeat_factor=args.repeat_factor,
            val_fraction=args.val_fraction,
            roi_predictions_dirs=tuple(args.roi_predictions_dir),
            roi_threshold=args.roi_threshold,
            roi_loss_weight=args.roi_loss_weight,
            background_loss_weight=args.background_loss_weight,
            min_roi_ratio=args.min_roi_ratio,
            lr_scheduler=args.lr_scheduler,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
            save_best=args.save_best,
            scenario_id=args.scenario_id,
            dataset_version=args.dataset_version,
            roi_model_version=args.roi_model_version,
            feature_ae_version=args.feature_ae_version,
            run_name=args.run_name,
            metric_eval_manifest_path=args.metric_eval_manifest,
            metric_eval_device=args.metric_eval_device,
            metric_eval_roi_predictions_dirs=tuple(args.metric_eval_roi_predictions_dir),
            gt_masks_manifest=args.gt_masks_manifest,
            validation_set_id=args.validation_set_id,
            metric_eval_every_epochs=args.metric_eval_every_epochs,
            metric_eval_start_epoch=args.metric_eval_start_epoch,
            metric_eval_batch_size=args.metric_eval_batch_size,
            metric_eval_tile_stride=args.metric_eval_tile_stride,
            metric_eval_layer_weights=parse_layer_loss_weights(args.metric_eval_layer_weights),
            metric_eval_calibrate_normal=args.metric_eval_calibrate_normal,
            metric_eval_calibration_mode=args.metric_eval_calibration_mode,
            metric_eval_calibration_stat=args.metric_eval_calibration_stat,
            metric_eval_calibration_max_images=args.metric_eval_calibration_max_images,
            metric_eval_score_region=args.metric_eval_score_region,
            metric_eval_apply_score_region_to_map=args.metric_eval_apply_score_region_to_map,
            metric_eval_score_smoothing=args.metric_eval_score_smoothing,
            metric_eval_score_image=args.metric_eval_score_image,
            metric_eval_topk_fraction=args.metric_eval_topk_fraction,
            metric_eval_save_score_maps=args.metric_eval_save_score_maps,
            metric_eval_save_previews=args.metric_eval_save_previews,
            metric_eval_max_previews=args.metric_eval_max_previews,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
