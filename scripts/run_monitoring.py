"""Run the IQA monitoring batch boundary.

This is the ``evaluate_lifecycle_conditions`` task of ``iqa_monitoring`` run as a
container on the data image (ADR 0008, issue 13): it evaluates the data-event
lifecycle rule **and** the monitoring thresholds (``configs/monitoring_thresholds.yaml``)
**inside the data image** -- no ``iqa`` import in the Airflow scheduler -- and
prints the result as JSON (the container stdout is the task's XCom).

The threshold evaluation is real: the candidate ``roi_fail_rate`` is compared
against the quality warning/critical thresholds from the config in-container.
Pushing the resulting metrics to Prometheus/Grafana is runtime observability,
tracked separately (issue 23).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iqa.monitoring import LifecycleSignal, evaluate_lifecycle_signal, should_trigger_lifecycle
from scripts.airflow_contracts import load_yaml_config, print_json, str2bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--conforming-validated-count", type=int, default=0)
    # Passed as a value (not a flag) so it survives templated argv from Airflow.
    parser.add_argument("--drift-confirmed", type=str2bool, default=False)
    parser.add_argument("--roi-fail-rate", type=float, default=0.0)
    parser.add_argument(
        "--thresholds-config",
        type=Path,
        default=Path("configs/monitoring_thresholds.yaml"),
    )
    return parser.parse_args()


def _evaluate_roi_fail_rate(roi_fail_rate: float, thresholds: dict[str, object]) -> dict[str, object]:
    """Compare the candidate ROI fail rate against the quality thresholds."""
    quality = thresholds.get("quality", {}) if isinstance(thresholds, dict) else {}
    warning = quality.get("roi_fail_rate_warning")
    critical = quality.get("roi_fail_rate_critical")
    status = "ok"
    if critical is not None and roi_fail_rate >= critical:
        status = "critical"
    elif warning is not None and roi_fail_rate >= warning:
        status = "warning"
    return {
        "metric": "roi_fail_rate",
        "value": roi_fail_rate,
        "warning": warning,
        "critical": critical,
        "status": status,
        "breached": status != "ok",
    }


def main() -> None:
    args = parse_args()
    signal = LifecycleSignal(
        scenario_id=args.scenario_id,
        conforming_validated_count=args.conforming_validated_count,
        drift_confirmed=args.drift_confirmed,
        roi_fail_rate=args.roi_fail_rate,
    )
    decision = evaluate_lifecycle_signal(signal)
    thresholds = load_yaml_config(args.thresholds_config)
    roi_eval = _evaluate_roi_fail_rate(args.roi_fail_rate, thresholds)
    result = {
        "service": "iqa-monitoring",
        "signal": signal.to_dict(),
        "lifecycle_decision": decision.to_dict(),
        "trigger_lifecycle": should_trigger_lifecycle(signal),
        # Real in-container threshold evaluation (configs/monitoring_thresholds.yaml).
        "thresholds_evaluated": bool(thresholds),
        "roi_fail_rate_evaluation": roi_eval,
        "status": "validated",
    }
    print_json(result)


if __name__ == "__main__":
    main()
