"""Run the IQA ingestion batch boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/casting_piece_events.csv"))
    parser.add_argument("--source", choices=["historical_replay", "production_ingest"], default="historical_replay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "service": "iqa-ingestion",
        "source": args.source,
        "manifest": str(args.manifest),
        "status": "planned",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
