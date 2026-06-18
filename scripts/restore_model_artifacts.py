"""Restore model checkpoints from manifests into the local IQA model cache."""

from __future__ import annotations

import argparse
import json

from iqa.models.artifacts import (
    DEFAULT_FEATURE_AE_MODEL_VERSION,
    DEFAULT_ROI_MODEL_VERSION,
    resolve_model_checkpoint,
)

KNOWN_MODEL_VERSIONS = (
    DEFAULT_ROI_MODEL_VERSION,
    DEFAULT_FEATURE_AE_MODEL_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model-version", action="append", dest="model_versions")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--strict-checksum", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_versions = KNOWN_MODEL_VERSIONS if args.all else tuple(args.model_versions)
    restored = []
    for model_version in model_versions:
        path = resolve_model_checkpoint(model_version, strict_checksum=args.strict_checksum)
        restored.append({"model_version": model_version, "checkpoint": str(path)})
    print(json.dumps({"restored": restored}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
