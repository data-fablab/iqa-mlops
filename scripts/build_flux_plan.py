"""Validate presence of replay metadata files.

The full CSV regeneration depends on restored project data. This script keeps
the public command available and fails with a clear message when inputs are
missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = [
    "casting_images_inventory.csv",
    "casting_piece_events.csv",
    "feature_ae_bootstrap_events.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [name for name in REQUIRED if not (args.metadata_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing metadata CSV(s): {', '.join(missing)}")
    print("Replay plan inputs are present.")


if __name__ == "__main__":
    main()
