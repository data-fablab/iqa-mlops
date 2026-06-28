"""Bake offline warm-start checkpoints for class2 and class3 (Issue 26).

Trains a Feature-AE from the multiclass manifest (feature_ae_good_v003) using
the stable_base checkpoint as init, with enough epochs to produce a checkpoint
that reliably promotes in a subsequent 1-epoch warm-started cycle.

Usage (GPU required):
    python -m scripts.bake_warmstart_checkpoints \
        --image-root data/raw/hss-iad \
        --device cuda

Each baked checkpoint lands at the path referenced by
``configs/demo_warmstart_checkpoints.yaml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
    resolve_feature_ae_checkpoint,
)
from iqa.training.feature_ae import FeatureAETrainingConfig, train_feature_ae
from iqa.training.feature_ae_evaluation import parse_layer_loss_weights

MULTICLASS_MANIFEST = Path("data/model_datasets/feature_ae_good_v003.csv")
VALIDATION_MANIFEST = Path("data/validation/validation_set_v001.csv")
VALIDATION_GT_MASKS = Path("data/validation/validation_gt_masks_v001.csv")

TARGETS = {
    "class2": {
        "output_dir": Path(".cache/iqa/models/rd_feature_ae_class2_precuit"),
        "run_name": "bake_warmstart_class2",
    },
    "class3": {
        "output_dir": Path(".cache/iqa/models/rd_feature_ae_class3_precuit"),
        "run_name": "bake_warmstart_class3",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["class2", "class3"],
        choices=["class2", "class3"],
    )
    parser.add_argument("--no-init-checkpoint", action="store_true",
                        help="train from scratch instead of warm-starting from stable_base")
    return parser.parse_args()


def bake_checkpoint(
    target_key: str,
    *,
    image_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    init_checkpoint: Path | None,
) -> dict:
    target = TARGETS[target_key]
    output_dir = target["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "checkpoint.pt"

    config = FeatureAETrainingConfig(
        manifest_path=MULTICLASS_MANIFEST,
        image_root=image_root,
        output_checkpoint=output_checkpoint,
        scenario_id="bake_warmstart",
        dataset_version="feature_ae_good_v003",
        candidate_version=f"rd_feature_ae_{target_key}_precuit",
        roi_model_version=DEFAULT_ROI_MODEL_VERSION,
        feature_ae_version=DEFAULT_FEATURE_AE_MODEL_VERSION,
        run_name=target["run_name"],
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        initial_checkpoint_path=init_checkpoint,
        initial_checkpoint_policy="fresh" if init_checkpoint is None else "explicit",
        save_best=True,
        metric_eval_manifest_path=VALIDATION_MANIFEST,
        gt_masks_manifest=VALIDATION_GT_MASKS,
        metric_eval_device=device,
        metric_eval_every_epochs=1,
        # Run the (expensive) AUPIMO validation pass only on the final epoch
        # instead of every epoch: a warm-start precuit is just an init checkpoint,
        # so one end-of-training metric is enough and keeps the demo bake short.
        metric_eval_start_epoch=epochs,
        metric_eval_calibrate_normal=False,
        metric_eval_layer_weights={"layer2": 0.65, "layer3": 0.35},
        metric_eval_apply_score_region_to_map=True,
        require_business_metric_for_early_stopping=False,
    )

    print(f"\n{'='*60}")
    print(f"Baking warm-start checkpoint: {target_key}")
    print(f"  output: {output_checkpoint}")
    print(f"  epochs: {epochs}")
    print(f"  init:   {init_checkpoint or 'fresh'}")
    print(f"{'='*60}\n")

    result = train_feature_ae(config)
    print(f"\n{target_key} training complete:")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()

    init_checkpoint = None
    if not args.no_init_checkpoint:
        init_checkpoint = resolve_feature_ae_checkpoint(
            DEFAULT_FEATURE_AE_MODEL_VERSION, strict_checksum=True,
        )
        print(f"Using stable_base init checkpoint: {init_checkpoint}")

    results = {}
    for cls in args.classes:
        results[cls] = bake_checkpoint(
            cls,
            image_root=args.image_root,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            init_checkpoint=init_checkpoint,
        )

    print("\n" + "=" * 60)
    print("BAKE SUMMARY")
    print("=" * 60)
    for cls, result in results.items():
        target = TARGETS[cls]
        ckpt = target["output_dir"] / "checkpoint.pt"
        exists = ckpt.is_file()
        print(f"  {cls}: {ckpt} (exists={exists})")
    print()
    print("Next step: verify warm-start promotion with:")
    print("  python -m scripts.verify_warmstart_promotion --image-root data/raw/hss-iad --device cuda")


if __name__ == "__main__":
    main()
