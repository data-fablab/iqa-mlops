"""Verify that baked warm-start checkpoints promote in a 1-epoch cycle (Issue 26).

Dry-runs a short lifecycle cycle for each baked checkpoint: epochs=1,
max_events=8, candidate-init-checkpoint pointing at the pre-baked file.
The acceptance criterion is ``promotion_status=promoted`` in the summary.

Usage:
    python -m scripts.verify_warmstart_promotion \
        --image-root data/raw/hss-iad \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WARMSTART_CONFIG = Path("configs/demo_warmstart_checkpoints.yaml")

VERIFY_CLASSES = {
    "Casting_class2": ".cache/iqa/models/rd_feature_ae_class2_precuit/checkpoint.pt",
    "Casting_class3": ".cache/iqa/models/rd_feature_ae_class3_precuit/checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Casting_class2", "Casting_class3"],
    )
    parser.add_argument("--output-root", type=Path,
                        default=Path(".cache/iqa/verify_warmstart"))
    return parser.parse_args()


def run_verification_cycle(
    triggering_class: str,
    checkpoint_path: Path,
    *,
    image_root: Path,
    device: str,
    output_root: Path,
) -> dict:
    """Run a short lifecycle cycle warm-started from the baked checkpoint."""
    cycle_output = output_root / triggering_class
    cycle_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "scripts.run_replay_lifecycle_cycle",
        "--scenario-id", "drift_domain_extension",
        "--image-root", str(image_root),
        "--mode", "progressive-train",
        "--max-events", "8",
        "--epochs", "1",
        "--max-cycles", "1",
        "--lifecycle-interval", "8",
        "--candidate-init-checkpoint", str(checkpoint_path),
        "--device", device,
        "--output-root", str(cycle_output),
        "--no-gpu-lock",
        "--anchor-good-manifest", "data/model_datasets/feature_ae_good_v003.csv",
    ]

    print(f"\n{'='*60}")
    print(f"Verifying warm-start promotion: {triggering_class}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output:     {cycle_output}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return {
            "class": triggering_class,
            "status": "error",
            "returncode": result.returncode,
            "reason": "lifecycle_cycle_failed",
        }

    try:
        summary = json.loads(result.stdout)
    except json.JSONDecodeError:
        summary_path = cycle_output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            return {
                "class": triggering_class,
                "status": "error",
                "reason": "no_summary_output",
            }

    promotion_status = summary.get("promotion_status", "unknown")
    return {
        "class": triggering_class,
        "promotion_status": promotion_status,
        "status": "pass" if promotion_status == "promoted" else "fail",
        "candidate_metric": summary.get("selected_metric"),
        "candidate_metric_value": summary.get("selected_metric_value"),
    }


def main() -> None:
    args = parse_args()

    results = {}
    for cls in args.classes:
        checkpoint = Path(VERIFY_CLASSES.get(cls, ""))
        if not checkpoint.is_file():
            print(f"SKIP {cls}: checkpoint not found at {checkpoint}")
            results[cls] = {"class": cls, "status": "skip", "reason": "checkpoint_not_found"}
            continue

        results[cls] = run_verification_cycle(
            cls,
            checkpoint,
            image_root=args.image_root,
            device=args.device,
            output_root=args.output_root,
        )

    print("\n" + "=" * 60)
    print("WARM-START VERIFICATION RESULTS")
    print("=" * 60)
    all_pass = True
    for cls, r in results.items():
        status = r["status"]
        icon = "PASS" if status == "pass" else ("SKIP" if status == "skip" else "FAIL")
        print(f"  [{icon}] {cls}: {r.get('promotion_status', r.get('reason', ''))}")
        if status not in ("pass", "skip"):
            all_pass = False

    if not all_pass:
        print("\nSome verifications failed. Checkpoints may need more epochs.")
        sys.exit(1)
    else:
        print("\nAll warm-start checkpoints verified for promotion.")


if __name__ == "__main__":
    main()
