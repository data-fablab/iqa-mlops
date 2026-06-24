"""Validate the materialized Feature-AE MVP candidate manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/model_datasets")
MVP_DATASET_VERSION = "feature_ae_good_mvp_v001"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = args.output_dir / f"{MVP_DATASET_VERSION}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Feature-AE MVP manifest is missing: {manifest}")
    print(f"Validated {manifest}.")


if __name__ == "__main__":
    main()
