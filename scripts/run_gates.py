"""Run the IQA promotion-gates boundary.

This is the ``gates`` task of ``iqa_lifecycle`` run as a container (ADR 0008,
issue 10). It evaluates ``configs/promotion_gates.yaml`` against the candidate
metrics **inside the container** and **blocks** the DAG (non-zero exit) when any
gate fails, so registration/promotion never runs on a failing candidate.

The gate logic and the blocking behaviour are real. The candidate metrics arrive
as explicit args for now: the real metric flow from a real ``eval`` (via XCom
references) is wired with the train/eval runtime (issue 20).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from iqa.promotion.gates import evaluate_promotion_gates
from scripts.airflow_contracts import load_yaml_config, print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--recall", type=float, default=1.0)
    parser.add_argument("--ap", type=float, default=0.0)
    parser.add_argument("--orange-rate", type=float, default=0.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--prod-ap", type=float, default=None)
    parser.add_argument("--gates-config", type=Path, default=Path("configs/promotion_gates.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gates = evaluate_promotion_gates(
        candidate_recall=args.recall,
        candidate_ap=args.ap,
        candidate_orange_rate=args.orange_rate,
        candidate_latency_ms=args.latency_ms,
        prod_ap=args.prod_ap,
        gates_config=load_yaml_config(args.gates_config),
    )
    all_passed = bool(gates["all_passed"])
    print_json(
        {
            "service": "iqa-gates",
            "scenario_id": args.scenario_id,
            "gates": gates,
            "all_passed": all_passed,
            "status": "validated" if all_passed else "blocked",
        }
    )
    if not all_passed:
        # Non-zero exit fails the Airflow task -> downstream mlflow/promotion skip.
        print("iqa-gates: promotion blocked (a gate failed)", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
