"""Collect and journal durable lifecycle signals from PostgreSQL."""

from __future__ import annotations

import argparse

from iqa.metadata.repository import (
    POSTGRES_BACKEND,
    create_metadata_repository,
    metadata_backend,
)
from iqa.monitoring import collect_and_record_lifecycle_signal
from scripts.airflow_contracts import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--roi-window-size", type=int, default=100)
    parser.add_argument("--min-natural-conforming", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if metadata_backend() != POSTGRES_BACKEND:
        raise RuntimeError(
            "IQA_METADATA_BACKEND=postgres is required for durable lifecycle signals."
        )

    result = collect_and_record_lifecycle_signal(
        create_metadata_repository(),
        scenario_id=args.scenario_id,
        roi_window_size=args.roi_window_size,
        min_natural_conforming=args.min_natural_conforming,
    )
    print_json(result)


if __name__ == "__main__":
    main()
