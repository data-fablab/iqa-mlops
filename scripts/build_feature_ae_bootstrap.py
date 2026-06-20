"""Build and publish the initial Feature-AE bootstrap checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.models.artifacts import DEFAULT_ROI_MODEL_VERSION, resolve_roi_segmenter_checkpoint
from iqa.roi.bootstrap import generate_bootstrap_roi_predictions
from iqa.training.bootstrap import (
    BOOTSTRAP_ARTIFACT_URI,
    BOOTSTRAP_MODEL_VERSION,
    materialize_bootstrap_checkpoint,
    select_bootstrap_champion,
    update_bootstrap_manifest,
    upload_checkpoint_to_s3,
)
from iqa.training.feature_ae import FeatureAETrainingConfig, train_feature_ae
from iqa.training.feature_ae_contracts import (
    CANONICAL_FEATURE_AE_PREPROCESSING,
    FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
    canonical_feature_ae_preprocessing_dict,
)
from iqa.training.feature_ae_evaluation import parse_layer_loss_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/feature_ae_bootstrap_events.csv"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("data/validation/validation_set_v001.csv"))
    parser.add_argument("--gt-masks-manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path(".cache/iqa/models/rd_feature_ae_gated_v001_bootstrap/bootstrap_run"))
    parser.add_argument("--existing-run-dir", type=Path)
    parser.add_argument("--roi-output-dir", type=Path, default=Path("data/processed/roi/bootstrap_v001"))
    parser.add_argument("--manifest-output", type=Path, default=Path("models/manifests/rd_feature_ae_gated_v001_bootstrap/model_manifest.json"))
    parser.add_argument("--artifact-uri", default=BOOTSTRAP_ARTIFACT_URI)
    parser.add_argument("--roi-model-version", default=DEFAULT_ROI_MODEL_VERSION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--metric-eval-every-epochs", type=int, default=1)
    parser.add_argument("--metric-eval-start-epoch", type=int, default=1)
    parser.add_argument("--metric-eval-batch-size", type=int, default=8)
    parser.add_argument("--metric-early-stopping-patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--layer-loss-weights", nargs="*", default=["layer2=0.65", "layer3=0.35"])
    parser.add_argument("--force-roi", action="store_true")
    parser.add_argument("--skip-roi", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish-minio", action="store_true")
    parser.add_argument(
        "--allow-noncanonical-preprocessing",
        action="store_true",
        help="Only for local tests/dev; server bootstrap candidates must use the canonical Feature-AE preprocessing contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(_dry_run_plan(args), indent=2, sort_keys=True))
        return
    if args.image_root is None:
        raise ValueError("--image-root is required unless --dry-run is used.")

    run_dir = args.existing_run_dir or args.run_dir
    if args.existing_run_dir is None:
        _ensure_roi_predictions(args)
        _train_bootstrap(args)

    champion = select_bootstrap_champion(run_dir)
    canonical_checkpoint = materialize_bootstrap_checkpoint(champion, run_dir / "checkpoint.pt")
    if args.publish_minio:
        upload_checkpoint_to_s3(canonical_checkpoint, args.artifact_uri)
    manifest = update_bootstrap_manifest(
        args.manifest_output,
        champion,
        artifact_uri=args.artifact_uri,
        dataset_version="feature_ae_good_v001_bootstrap",
        validation_set_id="validation_set_v001",
        roi_model_version=args.roi_model_version,
    )
    print(
        json.dumps(
            {
                "artifact_uri": args.artifact_uri,
                "checkpoint": str(canonical_checkpoint),
                "manifest": str(args.manifest_output),
                "model_version": BOOTSTRAP_MODEL_VERSION,
                "published_minio": bool(args.publish_minio),
                "selected_epoch": champion.selected_epoch,
                "selected_metric": champion.selected_metric,
                "selected_metric_value": champion.selected_metric_value,
                "sha256": manifest["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _dry_run_plan(args: argparse.Namespace) -> dict[str, object]:
    return {
        "artifact_uri": args.artifact_uri,
        "device": args.device,
        "dry_run": True,
        "epochs": args.epochs,
        "manifest": str(args.manifest),
        "metric_eval_every_epochs": args.metric_eval_every_epochs,
        "metric_early_stopping_patience": args.metric_early_stopping_patience,
        "model_version": BOOTSTRAP_MODEL_VERSION,
        "preprocessing_contract": canonical_feature_ae_preprocessing_dict(),
        "preprocessing_contract_version": FEATURE_AE_PREPROCESSING_CONTRACT_VERSION,
        "publish_minio": bool(args.publish_minio),
        "ranking_policy": "pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap -> image_auroc; val_loss stability only",
        "roi_model_version": args.roi_model_version,
        "run_dir": str(args.existing_run_dir or args.run_dir),
        "validation_manifest": str(args.validation_manifest),
    }


def _ensure_roi_predictions(args: argparse.Namespace) -> None:
    roi_index = args.roi_output_dir / "roi_predictions.csv"
    if args.skip_roi or (roi_index.is_file() and not args.force_roi):
        return
    checkpoint = resolve_roi_segmenter_checkpoint(args.roi_model_version)
    generate_bootstrap_roi_predictions(
        manifest_path=args.manifest,
        image_root=args.image_root,
        checkpoint_path=checkpoint,
        output_dir=args.roi_output_dir,
        roi_model_version=args.roi_model_version,
        dataset_version="feature_ae_good_v001_bootstrap",
        scenario_id="bootstrap_v001",
        device=args.device,
    )


def _train_bootstrap(args: argparse.Namespace) -> dict[str, object]:
    return train_feature_ae(
        FeatureAETrainingConfig(
            manifest_path=args.manifest,
            image_root=args.image_root,
            output_checkpoint=args.run_dir / "checkpoint.pt",
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            layers=tuple(args.layers),
            layer_loss_weights=parse_layer_loss_weights(args.layer_loss_weights),
            roi_predictions_dirs=(args.roi_output_dir,),
            metric_eval_manifest_path=args.validation_manifest,
            metric_eval_roi_predictions_dirs=(args.roi_output_dir,),
            gt_masks_manifest=args.gt_masks_manifest,
            validation_set_id="validation_set_v001",
            metric_eval_every_epochs=args.metric_eval_every_epochs,
            metric_eval_start_epoch=args.metric_eval_start_epoch,
            metric_eval_batch_size=args.metric_eval_batch_size,
            metric_early_stopping_patience=args.metric_early_stopping_patience,
            require_business_metric_for_early_stopping=True,
            allow_noncanonical_preprocessing=args.allow_noncanonical_preprocessing,
            metric_eval_calibrate_normal=False,
            metric_eval_apply_score_region_to_map=False,
            metric_eval_score_region=CANONICAL_FEATURE_AE_PREPROCESSING.score_region,
            metric_eval_score_smoothing=CANONICAL_FEATURE_AE_PREPROCESSING.score_smoothing,
            metric_eval_score_image=CANONICAL_FEATURE_AE_PREPROCESSING.score_image,
            metric_eval_topk_fraction=CANONICAL_FEATURE_AE_PREPROCESSING.topk_fraction,
            roi_model_version=args.roi_model_version,
            feature_ae_version=BOOTSTRAP_MODEL_VERSION,
            scenario_id="bootstrap_v001",
            dataset_version="feature_ae_good_v001_bootstrap",
            manifest_version="feature_ae_bootstrap_events_v001",
            candidate_version=BOOTSTRAP_MODEL_VERSION,
            run_name="feature_ae_bootstrap_v001",
        )
    )


if __name__ == "__main__":
    main()
