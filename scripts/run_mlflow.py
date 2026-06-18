"""Run the IQA MLflow-registration boundary.

This is the ``mlflow`` task of ``iqa_lifecycle`` run as a container on the ``ml``
image (ADR 0008, issue 10). It resolves the **scenario-isolated** registered model
name (ADR 0006: ``feature_ae__<scenario_id>``) and reports the registry reference.

Scope note: this is a ``validated-summary`` boundary -- the scenario isolation of
the name is real, but it does NOT yet register the run in the MLflow Registry
(``registered: false``). Real registration needs a real training run_id and an
MLflow server; it is tracked separately (issue 21), mirroring the earlier splits
(issues 18, 19, 20).
"""

from __future__ import annotations

import argparse

from iqa.registry import ModelRegistryRef, registered_model_name
from scripts.airflow_contracts import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--stage", default="candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = registered_model_name(args.scenario_id)
    registry_ref = ModelRegistryRef(
        scenario_id=args.scenario_id,
        registered_model_name=name,
        stage=args.stage,
    )
    print_json(
        {
            "service": "iqa-mlflow",
            "registered_model_name": name,
            "stage": args.stage,
            "run_id": args.run_id or None,
            "registry": registry_ref.to_dict(),
            # Real MLflow Registry registration is deferred to issue 21.
            "registered": False,
            "status": "validated",
        }
    )


if __name__ == "__main__":
    main()
