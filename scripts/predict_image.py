"""Run a Feature-AE prediction on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.inference import predict_feature_ae_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--feature-ae-checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384, help="Local Feature-AE tile size.")
    parser.add_argument("--context-size", type=int, default=768, help="Context image size for tiled_context mode.")
    parser.add_argument("--preprocessing-mode", choices=["letterbox", "tiled_context"], default="tiled_context")
    parser.add_argument("--threshold-orange", type=float, default=0.02)
    parser.add_argument("--threshold-red", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained-teacher", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_feature_ae_image(
        args.image,
        args.feature_ae_checkpoint,
        image_size=args.image_size,
        context_size=args.context_size,
        preprocessing_mode=args.preprocessing_mode,
        threshold_orange=args.threshold_orange,
        threshold_red=args.threshold_red,
        device=args.device,
        pretrained_teacher=not args.no_pretrained_teacher,
    )
    print(json.dumps(prediction.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
