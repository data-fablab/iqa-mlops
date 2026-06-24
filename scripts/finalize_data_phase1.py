"""Validate IQA canonical lightweight data manifests.

The original Phase 1 generator is retired for the MVP lifecycle: validation,
calibration and replay partitions are now materialized explicitly under versioned
CSV files. This command remains as a compatibility boundary for DVC/Airflow
contracts and fails clearly if a canonical manifest is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CANONICAL_MANIFESTS = (
    Path("data/metadata/casting_piece_events.csv"),
    Path("data/metadata/casting_images_inventory.csv"),
    Path("data/metadata/feature_ae_bootstrap_events.csv"),
    Path("data/metadata/casting_flux_replay_plan_natural_v003.csv"),
    Path("data/metadata/casting_flux_replay_plan_drift.csv"),
    Path("data/validation/validation_set_replay_representative_v001.csv"),
    Path("data/validation/calibration_good_reference_v001.csv"),
    Path("data/model_datasets/feature_ae_good_mvp_v001.csv"),
    Path("data/metadata/feature_ae_partition_report_v001.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [str(path) for path in CANONICAL_MANIFESTS if not (args.root / path).is_file()]
    if missing:
        raise SystemExit(f"Missing canonical data manifest(s): {', '.join(missing)}")
    print("Canonical IQA data manifests are present.")


if __name__ == "__main__":
    main()
