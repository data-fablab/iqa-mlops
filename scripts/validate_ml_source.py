"""Validate that the reproducible ML source path is present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PATHS = [
    Path("src/iqa/ingestion"),
    Path("src/iqa/datasets"),
    Path("src/iqa/training"),
    Path("src/iqa/inference"),
    Path("src/iqa/models/feature_ae"),
    Path("src/iqa/models/segmentation"),
    Path("data/metadata/feature_ae_bootstrap_events.csv"),
    Path("data/metadata/casting_piece_events.csv"),
    Path("data/metadata/replay_scenarios.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in REQUIRED_PATHS if not (args.root / path).exists()]
    result = {
        "ok": not missing,
        "missing": missing,
        "checked": [str(path) for path in REQUIRED_PATHS],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
