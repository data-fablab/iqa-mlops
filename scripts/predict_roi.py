"""Run the fixed ROI segmenter on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.inference import predict_roi_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-roi-ratio", type=float, default=0.01)
    parser.add_argument("--max-roi-ratio", type=float, default=0.98)
    parser.add_argument("--surface-class", type=int, default=1)
    parser.add_argument("--mask-mode", choices=("argmax", "threshold"), default="argmax")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-mask", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = predict_roi_image(
        args.image,
        args.checkpoint,
        image_size=args.image_size,
        context_size=args.context_size,
        threshold=args.threshold,
        min_roi_ratio=args.min_roi_ratio,
        max_roi_ratio=args.max_roi_ratio,
        surface_class=args.surface_class,
        mask_mode=args.mask_mode,
        device=args.device,
        output_mask=args.output_mask,
    )
    print(json.dumps(prediction.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
