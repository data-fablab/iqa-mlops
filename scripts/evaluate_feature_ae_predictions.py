"""Recompute Feature-AE business metrics from a materialized predictions.npz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.training.feature_ae_evaluation import evaluate_feature_ae_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-orange", type=float, required=True)
    parser.add_argument("--threshold-red", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_feature_ae_predictions(
        args.predictions,
        threshold_orange=args.threshold_orange,
        threshold_red=args.threshold_red,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "metrics.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"metrics": output_path, "predictions": args.predictions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
