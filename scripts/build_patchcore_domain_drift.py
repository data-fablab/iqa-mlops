"""Build + register the PatchCore domain-drift detector (Issue 11).

Runs **inside the GPU ``iqa-inference`` container** (the only place with CUDA torch +
the cached WideResNet50 weights). It:

1. builds the coreset memory bank from ``Casting_class1/train/good`` minus a hold-out,
2. calibrates the per-piece out-of-domain threshold = p90 of the class1 hold-out,
3. persists the detector (``memory_bank.pt`` / ``calibration.yaml`` /
   ``model_manifest.json``) to ``--output-dir`` (container ``/tmp``; then ``docker cp``
   to the host ``.cache/iqa/models/patchcore_domain_drift_v001`` which is mounted RO),
4. measures the **domain separation** (median + rate@class1_p90 per class) on
   class1 hold-out / class2 / class3 *good* images (domain shift, not defects),
5. logs an MLflow run in the ``iqa-domain-drift`` experiment: params + separation
   metrics + the per-class score table as a MinIO artifact.

Run (from the repo root, stack up)::

    docker exec deploy-iqa-inference-1 python -m scripts.build_patchcore_domain_drift \
        --output-dir /tmp/patchcore_domain_drift_v001
    docker cp deploy-iqa-inference-1:/tmp/patchcore_domain_drift_v001 \
        .cache/iqa/models/patchcore_domain_drift_v001
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np

from iqa.inference.domain_drift import (
    DEFAULT_CORESET_PATCHES,
    DEFAULT_MEM_IMAGES,
    DEFAULT_PERCENTILE,
    DEFAULT_SEED,
    MODEL_VERSION,
    PatchCoreDomainDriftDetector,
    balanced_pool,
    regime_for_score,
)

DEFAULT_DATASET_ROOT = "/opt/iqa/iqa-mlops/data/raw/hss-iad"
EXPERIMENT_NAME = "iqa-domain-drift"
# Good images only: we measure a DOMAIN shift across products, not defects.
CLASS_GLOBS = {
    "class1": "Casting_class1/train/good/*.jpg",
    "class2": "Casting_class2/train/good/*.jpg",
    "class3": "Casting_class3/train/good/*.jpg",
}


def log(message: str) -> None:
    print(f"[build-patchcore] {message}", flush=True)


def _list_images(dataset_root: str, pattern: str) -> list[str]:
    return sorted(glob.glob(str(Path(dataset_root) / pattern)))


def build_and_evaluate(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    cover_classes: list[str] = list(args.cover_classes)

    # Collect images for each covered class, split holdout vs bank pool.
    bank_images_by_class: dict[str, list[str]] = {}
    all_holdout: list[str] = []
    for cls_key in cover_classes:
        glob_key = cls_key.replace("Casting_", "")
        pattern = CLASS_GLOBS.get(glob_key)
        if pattern is None:
            raise SystemExit(f"no glob pattern for class '{cls_key}' (tried key '{glob_key}')")
        images = _list_images(args.dataset_root, pattern)
        if len(images) < args.holdout + 10:
            raise SystemExit(f"not enough {cls_key} images ({len(images)}) for holdout {args.holdout}")
        pool = list(images)
        random.shuffle(pool)
        holdout = pool[: args.holdout]
        bank_pool = pool[args.holdout :]
        all_holdout.extend(holdout)
        bank_images_by_class[cls_key] = bank_pool
        log(f"{cls_key}: total {len(images)}, holdout {len(holdout)}, bank pool {len(bank_pool)}")

    mem_paths = balanced_pool(bank_images_by_class, args.mem_images, seed=args.seed)
    log(f"balanced bank pool: {len(mem_paths)} images across {len(cover_classes)} class(es)")

    detector = PatchCoreDomainDriftDetector(
        device=args.device, seed=args.seed, coreset_patches=args.coreset,
        covered_classes=cover_classes,
    )
    log("building memory bank (WRN50 layer2+3, coreset)...")
    bank = detector.build_bank(mem_paths)
    log(f"bank shape {tuple(bank.shape)}")

    log(f"calibrating per-piece threshold = p{args.percentile:g} of union hold-out ({len(all_holdout)} images)...")
    calibration = detector.calibrate(all_holdout, percentile=args.percentile)
    threshold = calibration.threshold
    log(f"threshold = {threshold:.4f} (holdout median {calibration.class1_score_median:.4f})")

    # Separation metrics on good images per class.
    eval_paths: dict[str, list[str]] = {}
    for cls_key in cover_classes:
        glob_key = cls_key.replace("Casting_", "")
        eval_paths[glob_key] = _list_images(args.dataset_root, CLASS_GLOBS[glob_key])
        random.shuffle(eval_paths[glob_key])
        eval_paths[glob_key] = eval_paths[glob_key][: args.per_class_eval]
    for klass in ("class1", "class2", "class3"):
        if klass not in eval_paths:
            paths = _list_images(args.dataset_root, CLASS_GLOBS[klass])
            random.shuffle(paths)
            eval_paths[klass] = paths[: args.per_class_eval]

    rows: list[tuple[str, str, float]] = []
    summary: dict[str, dict[str, float]] = {}
    for klass, paths in eval_paths.items():
        scores = [detector.score(p) for p in paths]
        for path, score in zip(paths, scores):
            rows.append((klass, path, score))
        arr = np.asarray(scores, dtype=np.float64)
        rate = float(np.mean(arr >= threshold)) if arr.size else 0.0
        summary[klass] = {
            "count": int(arr.size),
            "median": float(np.median(arr)) if arr.size else 0.0,
            "p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
            "rate_at_threshold": rate,
        }
        log(
            f"{klass:<7} n={summary[klass]['count']:>3} "
            f"median={summary[klass]['median']:.3f} rate@threshold={rate:.2f}"
        )

    output_dir = Path(args.output_dir)
    detector.save(output_dir)
    log(f"detector saved to {output_dir} ({MODEL_VERSION})")

    score_table = output_dir / "class_scores.csv"
    score_table.write_text(
        "class,image,score\n" + "\n".join(f"{k},{p},{s:.6f}" for k, p, s in rows) + "\n",
        encoding="utf-8",
    )

    return {
        "detector": detector,
        "calibration": calibration,
        "summary": summary,
        "output_dir": output_dir,
        "score_table": score_table,
        "mem_images": len(mem_paths),
        "covered_classes": cover_classes,
    }


def log_mlflow(args: argparse.Namespace, result: dict) -> None:
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    calibration = result["calibration"]
    summary = result["summary"]
    with mlflow.start_run(run_name=f"{MODEL_VERSION}_build"):
        mlflow.set_tags(
            {
                "model_version": MODEL_VERSION,
                "signal": "domain_drift",
                "detector": "patchcore_distance_to_nominal",
                "purpose": "domain_drift_only_not_defect_detection",
            }
        )
        mlflow.log_params(
            {
                "backbone": "wide_resnet50_2",
                "feature_layers": "layer2+layer3",
                "mem_images": result["mem_images"],
                "coreset_patches": args.coreset,
                "seed": args.seed,
                "percentile": args.percentile,
                "threshold_per_piece": round(calibration.threshold, 6),
                "covered_classes": ",".join(result.get("covered_classes", ["Casting_class1"])),
            }
        )
        mlflow.log_metric("threshold_per_piece", calibration.threshold)
        for klass, stats in summary.items():
            mlflow.log_metric(f"{klass}_median", stats["median"])
            mlflow.log_metric(f"{klass}_rate_at_threshold", stats.get("rate_at_threshold", stats.get("rate_at_class1_p90", 0.0)))
        mlflow.log_artifact(str(result["score_table"]), artifact_path="separation")
        mlflow.log_artifact(str(result["output_dir"] / "calibration.yaml"), artifact_path="detector")
        mlflow.log_artifact(str(result["output_dir"] / "model_manifest.json"), artifact_path="detector")
        run_id = mlflow.active_run().info.run_id
    log(f"MLflow run logged in '{EXPERIMENT_NAME}' (run_id={run_id})")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default=f"/tmp/{MODEL_VERSION}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mem-images", type=int, default=DEFAULT_MEM_IMAGES)
    parser.add_argument("--coreset", type=int, default=DEFAULT_CORESET_PATCHES)
    parser.add_argument("--holdout", type=int, default=40, help="class1 images reserved for calibration")
    parser.add_argument("--per-class-eval", type=int, default=40, help="good images scored per class for separation")
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cover-classes", nargs="+", default=["Casting_class1"],
        help="classes to cover in the bank (e.g. Casting_class1 Casting_class2)",
    )
    parser.add_argument("--tracking-uri", default=None, help="MLflow URI (default: env MLFLOW_TRACKING_URI)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip the MLflow run (bank/calib only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tracking_uri is None:
        import os

        args.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    result = build_and_evaluate(args)
    if args.no_mlflow:
        log("MLflow logging skipped (--no-mlflow)")
    else:
        try:
            log_mlflow(args, result)
        except Exception as exc:  # noqa: BLE001 - the detector is already on disk
            log(f"WARNING: MLflow logging failed ({exc!r}); detector still saved on disk")
    summary = result["summary"]
    log(
        "separation summary: "
        + " ".join(f"{k}@thr={v['rate_at_threshold']:.2f}" for k, v in summary.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
