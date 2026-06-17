"""Run the IQA replay batch boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from iqa.replay import REPLAY_SCENARIOS, list_replay_scenarios
from scripts.airflow_contracts import csv_manifest_summary, print_json, read_csv_rows, stable_unique


KNOWN_SCENARIO_IDS = {scenario.scenario_id for scenario in REPLAY_SCENARIOS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--plan", type=Path, default=Path("data/metadata/casting_flux_replay_plan_natural.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenario_id not in KNOWN_SCENARIO_IDS:
        raise ValueError(f"unknown replay scenario_id: {args.scenario_id}")
    rows = [row for row in read_csv_rows(args.plan, label="replay plan") if row.get("scenario_id") == args.scenario_id]
    if not rows:
        raise ValueError(f"replay plan has no rows for scenario_id={args.scenario_id}: {args.plan}")
    summary = csv_manifest_summary(args.plan, label="replay plan")
    result = {
        "service": "iqa-replay",
        "scenario_id": args.scenario_id,
        "plan": summary,
        "plan_event_count": len(rows),
        "dataset_versions": stable_unique(row.get("dataset_version") for row in rows),
        "lot_ids": stable_unique(row.get("lot_id") for row in rows),
        "source_classes": stable_unique(row.get("source_class") for row in rows),
        "known_scenarios": list_replay_scenarios(),
        "status": "validated",
    }
    print_json(result)


if __name__ == "__main__":
    main()
