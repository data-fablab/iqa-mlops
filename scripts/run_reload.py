"""Run the IQA inference-reload boundary.

This is the ``reload`` task of ``iqa_lifecycle`` run as a container on the data
image (ADR 0008, issue 11). It closes the automated MLOps loop: after a prod
promotion, ``iqa-inference`` must reload so it serves the new version.

The skip rule is real: reload only runs for ``target_stage == prod`` -- a
promotion to ``test`` leaves production untouched, so the inference service is
not asked to reload. For a prod promotion it resolves the scenario-isolated
registered model name and reports the reload that should be triggered.

Scope note: this is a ``validated-summary`` boundary -- the skip rule and the
name resolution are real, but it does NOT yet call the ``iqa-inference`` reload
contract over HTTP (``reloaded: false``). Wiring the real reload (and asserting
the service then serves the new version) needs both services up; it is tracked
separately (issue 22), mirroring the earlier splits (issues 18-21).
"""

from __future__ import annotations

import argparse

from iqa.registry import registered_model_name
from scripts.airflow_contracts import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--target-stage", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_stage = args.target_stage
    if target_stage != "prod":
        print_json(
            {
                "service": "iqa-reload",
                "scenario_id": args.scenario_id,
                "target_stage": target_stage,
                "reloaded": False,
                "status": "skipped",
                "reason": (
                    f"target_stage is {target_stage}; the inference reload only "
                    "runs for prod promotions"
                ),
            }
        )
        return

    name = registered_model_name(args.scenario_id)
    print_json(
        {
            "service": "iqa-reload",
            "scenario_id": args.scenario_id,
            "registered_model_name": name,
            "target_stage": target_stage,
            # Real HTTP reload of iqa-inference is deferred to issue 22.
            "reloaded": False,
            "status": "validated",
        }
    )


if __name__ == "__main__":
    main()
