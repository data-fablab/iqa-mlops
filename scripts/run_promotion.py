"""Run the IQA promotion boundary.

This is the ``promotion`` task of ``iqa_lifecycle`` run as a container on the
``ml`` image (ADR 0008, issue 11). MLflow is the source of truth for promotion
(ADR 0006): this boundary resolves the **scenario-isolated** registered model
name (``feature_ae__<scenario_id>``) and the intended ``candidate -> target_stage``
transition, and reports whether a prod promotion would first snapshot the current
prod (rollback safety).

Scope note: this is a ``validated-summary`` boundary -- the name resolution and
the prod-snapshot rule are real, but it does NOT yet transition the model in the
MLflow Registry (``promoted: false``). The real Registry transition
(``promote_model_with_gates`` / ``save_previous_prod_before_promotion``) needs a
real registered version and an MLflow server; it is tracked separately (issue 22),
mirroring the earlier splits (issues 18-21).
"""

from __future__ import annotations

import argparse

from iqa.registry import ModelRegistryRef, registered_model_name
from scripts.airflow_contracts import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--source-stage", default="candidate")
    parser.add_argument("--target-stage", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = registered_model_name(args.scenario_id)
    target_stage = args.target_stage
    # Real rule (ADR 0006): a prod promotion first snapshots the current prod so a
    # rollback is always possible. Only prod transitions need this safeguard.
    snapshot_previous_prod = target_stage == "prod"
    registry_ref = ModelRegistryRef(
        scenario_id=args.scenario_id,
        registered_model_name=name,
        stage=target_stage,
    )
    print_json(
        {
            "service": "iqa-promotion",
            "scenario_id": args.scenario_id,
            "registered_model_name": name,
            "transition": {"from": args.source_stage, "to": target_stage},
            "snapshot_previous_prod": snapshot_previous_prod,
            "registry": registry_ref.to_dict(),
            # Real MLflow Registry transition is deferred to issue 22.
            "promoted": False,
            "status": "validated",
        }
    )


if __name__ == "__main__":
    main()
