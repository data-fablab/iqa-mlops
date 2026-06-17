"""Run the IQA replay batch boundary.

This is the ``run_replay`` task of ``iqa_replay`` run as a container on the data
image (ADR 0008, issue 12): it validates the replay plan for the requested
scenario **inside the data image** -- no ``iqa`` import in the Airflow scheduler --
and reports the plan summary as JSON (the container stdout is the task's XCom:
references only, no payloads).

Replayed events keep their temporal/simulation semantics: each plan row carries
``event_time`` (when the original event happened), ``recorded_at`` (when it was
recorded) and ``is_simulated`` (replay marker). This boundary reports which of
those fields are preserved across the replayed rows so the contract is verifiable
without emitting events; real event emission into the ingestion store is runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iqa.replay import REPLAY_SCENARIOS, list_replay_scenarios
from scripts.airflow_contracts import csv_manifest_summary, print_json, read_csv_rows, stable_unique


KNOWN_SCENARIO_IDS = {scenario.scenario_id for scenario in REPLAY_SCENARIOS}

# Temporal + simulation semantics a replayed event must keep (acceptance criterion).
REPLAY_EVENT_SEMANTIC_FIELDS = ("event_time", "recorded_at", "is_simulated")


def _preserved_event_fields(rows: list[dict[str, str]]) -> list[str]:
    """Semantic fields present (non-empty) on every replayed row."""
    return [
        field
        for field in REPLAY_EVENT_SEMANTIC_FIELDS
        if rows and all(row.get(field) not in (None, "") for row in rows)
    ]


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
        # Replayed events keep event_time/recorded_at/is_simulated (acceptance criterion).
        "preserved_event_fields": _preserved_event_fields(rows),
        "is_simulated_values": stable_unique(row.get("is_simulated") for row in rows),
        "known_scenarios": list_replay_scenarios(),
        "status": "validated",
    }
    print_json(result)


if __name__ == "__main__":
    main()
