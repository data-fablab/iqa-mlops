"""Generate fixed ROI masks for the Feature-AE bootstrap dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.roi.bootstrap import generate_bootstrap_roi_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi-model-version", required=True)
    parser.add_argument("--dataset-version", default="bootstrap_v001")
    parser.add_argument("--scenario-id", default="bootstrap_v001")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = generate_bootstrap_roi_predictions(
        manifest_path=args.manifest,
        image_root=args.image_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        roi_model_version=args.roi_model_version,
        dataset_version=args.dataset_version,
        scenario_id=args.scenario_id,
        device=args.device,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "roi_predictions_csv": str(args.output_dir / "roi_predictions.csv"),
                "n_predictions": len(artifacts),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
