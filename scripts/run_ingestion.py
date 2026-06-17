"""Run the IQA ingestion batch boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.airflow_contracts import csv_manifest_summary, print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/casting_piece_events.csv"))
    parser.add_argument("--source", choices=["historical_replay", "production_ingest"], default="historical_replay")
    parser.add_argument("--scenario-id", default="raw_ingestion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = csv_manifest_summary(args.manifest, label="ingestion manifest")
    result = {
        "service": "iqa-ingestion",
        "source": args.source,
        "scenario_id": args.scenario_id,
        "manifest": manifest,
        "status": "validated",
    }
    print_json(result)


if __name__ == "__main__":
    main()
