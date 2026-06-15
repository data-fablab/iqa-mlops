"""Validate restored IQA MVP repository contracts."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "configs/paths.yaml",
    "configs/replay_scenarios.yaml",
    "src/iqa/api/main.py",
    "src/iqa/ingestion/schemas.py",
    "src/iqa/storage/uris.py",
    "docs/adr/0004-postgresql-comme-metadata-store.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [path for path in REQUIRED_PATHS if not (args.root / path).exists()]
    if missing:
        raise SystemExit(f"Missing required path(s): {', '.join(missing)}")
    print("IQA MVP repository contracts are present.")


if __name__ == "__main__":
    main()
