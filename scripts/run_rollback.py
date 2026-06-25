"""Run the IQA model rollback boundary (Issue 5).

Executes the **existing** rollback path (``iqa.promotion.rollback``) when a promoted
model regresses on a business metric: the ``IqaModelRegression`` alert
(Prometheus, Issue 5) fires, ``iqa_rollback_sensor`` reads the shared ``ALERTS``
series and triggers ``iqa_rollback``, whose container task is this CLI.

It resolves the scenario-isolated registered model name
(``feature_ae__<scenario_id>``), takes the **current prod** version as the faulty
one (unless ``--faulty-version`` overrides it), and calls ``rollback_model`` to
restore ``previous_prod`` to prod and archive the faulty version. MLflow is the
source of truth (ADR 0006); no restoration logic is re-implemented here.
"""

from __future__ import annotations

import argparse
import os

from iqa.promotion import resolve_model_artifacts, rollback_model
from iqa.registry import registered_model_name
from scripts.airflow_contracts import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument(
        "--faulty-version",
        default=None,
        help="Version to roll back from (default: the current prod version).",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI).",
    )
    return parser.parse_args()


def run_rollback(
    scenario_id: str,
    *,
    faulty_version: str | None = None,
    tracking_uri: str | None = None,
) -> dict:
    """Resolve the faulty prod version and run the existing rollback path."""
    name = registered_model_name(scenario_id)
    if faulty_version is None:
        prod = resolve_model_artifacts(name, stage="prod", tracking_uri=tracking_uri)
        faulty_version = prod["version"]

    result = rollback_model(name, faulty_version=faulty_version, tracking_uri=tracking_uri)
    return {
        "service": "iqa-rollback",
        "scenario_id": scenario_id,
        "registered_model_name": name,
        "faulty_version": str(faulty_version),
        "status": "rolled_back" if result.get("success") else "failed",
        **result,
    }


def main() -> None:
    args = parse_args()
    payload = run_rollback(
        args.scenario_id,
        faulty_version=args.faulty_version,
        tracking_uri=args.tracking_uri,
    )
    print_json(payload)
    if not payload.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
