"""Run the IQA monitoring batch boundary."""

from __future__ import annotations

import argparse
import json

from iqa.monitoring import LifecycleSignal, evaluate_lifecycle_signal, should_trigger_lifecycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--conforming-validated-count", type=int, default=0)
    parser.add_argument("--drift-confirmed", action="store_true")
    parser.add_argument("--roi-fail-rate", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal = LifecycleSignal(
        scenario_id=args.scenario_id,
        conforming_validated_count=args.conforming_validated_count,
        drift_confirmed=args.drift_confirmed,
        roi_fail_rate=args.roi_fail_rate,
    )
    decision = evaluate_lifecycle_signal(signal)
    result = {
        "service": "iqa-monitoring",
        "signal": signal.to_dict(),
        "lifecycle_decision": decision.to_dict(),
        "trigger_lifecycle": should_trigger_lifecycle(signal),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
