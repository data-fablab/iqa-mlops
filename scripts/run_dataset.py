"""Run the IQA candidate-dataset boundary.

This is the ``dataset`` task of ``iqa_lifecycle`` run as a container (ADR 0008,
issue 08): it validates the candidate manifest **inside the data image** and
prints a summary as JSON.

Scope note: this is a ``validated-summary`` boundary -- it does NOT yet
materialise the dataset in MinIO/PostgreSQL (``materialized: false``). The real
data-plane write is tracked separately (issue 19), mirroring the ingestion split
(issue 18). Until then the summary documents what *would* be materialised.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.airflow_contracts import csv_manifest_summary, print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/model_datasets/feature_ae_good_v002.csv"),
    )
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--candidate-version", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = csv_manifest_summary(args.manifest, label="dataset manifest")
    result = {
        "service": "iqa-dataset",
        "scenario_id": args.scenario_id,
        "candidate_version": args.candidate_version or None,
        "manifest": manifest,
        # Real MinIO/PostgreSQL materialisation is deferred to issue 19.
        "materialized": False,
        "status": "validated",
    }
    print_json(result)


if __name__ == "__main__":
    main()
