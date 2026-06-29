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
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from iqa.monitoring import LifecycleSignal, evaluate_lifecycle_signal, should_trigger_lifecycle
from scripts.airflow_contracts import load_yaml_config, print_json, str2bool


DEFAULT_DRIFT_THRESHOLDS = {
    "min_window_events": 30,
    "confirm_windows": 2,
    "domain_ratio_critical": 0.50,
    "alert_rate_critical": 0.50,
    "red_rate_critical": 0.20,
    "unexpected_red_rate_critical": 0.20,
    "roi_fail_rate_critical": 0.10,
    "oracle_fn_rate_critical": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--conforming-validated-count", type=int, default=0)
    # Passed as a value (not a flag) so it survives templated argv from Airflow.
    parser.add_argument("--drift-confirmed", type=str2bool, default=False)
    parser.add_argument("--roi-fail-rate", type=float, default=0.0)
    parser.add_argument("--source-domain", default="piece_a_p4")
    parser.add_argument("--window-events", type=int, default=0)
    parser.add_argument("--domain-ratio", type=float, default=0.0)
    parser.add_argument("--alert-rate", type=float, default=0.0)
    parser.add_argument("--red-rate", type=float, default=0.0)
    parser.add_argument("--unexpected-red-rate", type=float, default=0.0)
    parser.add_argument("--oracle-fn-rate", type=float, default=0.0)
    parser.add_argument("--critical-window-count", type=int, default=0)
    parser.add_argument("--api-url", default=os.getenv("IQA_API_URL", ""))
    parser.add_argument("--service-token", default=os.getenv("IQA_SERVICE_TOKEN", ""))
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


def _drift_thresholds(thresholds: dict[str, object]) -> dict[str, float | int]:
    drift = thresholds.get("drift", {}) if isinstance(thresholds, dict) else {}
    merged = dict(DEFAULT_DRIFT_THRESHOLDS)
    if isinstance(drift, dict):
        for key in DEFAULT_DRIFT_THRESHOLDS:
            if key in drift:
                merged[key] = drift[key]  # type: ignore[assignment]
    return merged


def _score_ratio(value: float, critical: float) -> float:
    if critical <= 0:
        return 0.0
    return max(0.0, min(value / critical, 1.0))


def evaluate_drift_metrics(
    *,
    scenario_id: str,
    window_events: int,
    domain_ratio: float,
    alert_rate: float,
    red_rate: float,
    roi_fail_rate: float,
    oracle_fn_rate: float,
    critical_window_count: int,
    unexpected_red_rate: float = 0.0,
    drift_confirmed: bool = False,
    thresholds: dict[str, object] | None = None,
) -> dict[str, object]:
    drift_thresholds = _drift_thresholds(thresholds)
    min_window_events = int(drift_thresholds["min_window_events"])
    confirm_windows = int(drift_thresholds["confirm_windows"])
    domain_ratio_critical = float(drift_thresholds["domain_ratio_critical"])
    alert_rate_critical = float(drift_thresholds["alert_rate_critical"])
    red_rate_critical = float(drift_thresholds["red_rate_critical"])
    unexpected_red_rate_critical = float(drift_thresholds.get("unexpected_red_rate_critical", red_rate_critical))
    roi_fail_rate_critical = float(drift_thresholds["roi_fail_rate_critical"])
    oracle_fn_rate_critical = float(drift_thresholds["oracle_fn_rate_critical"])
    enough_events = window_events >= min_window_events
    signals = {
        "domain_ratio": domain_ratio >= domain_ratio_critical,
        "alert_rate": alert_rate >= alert_rate_critical,
        "red_rate": red_rate >= red_rate_critical,
        "unexpected_red_rate": unexpected_red_rate >= unexpected_red_rate_critical,
        "roi_fail_rate": roi_fail_rate >= roi_fail_rate_critical,
        "oracle_fn_rate": oracle_fn_rate >= oracle_fn_rate_critical,
    }
    degradation_signals = {
        key: value
        for key, value in signals.items()
        if key in {"alert_rate", "red_rate", "unexpected_red_rate", "roi_fail_rate", "oracle_fn_rate"}
    }
    degradation_signal_count = sum(1 for breached in degradation_signals.values() if breached)
    domain_signal = signals["domain_ratio"]
    critical_window = enough_events and domain_signal and degradation_signal_count > 0
    next_critical_window_count = (critical_window_count + 1) if critical_window else 0
    drift_suspected = enough_events and domain_signal
    drift_confirmed = bool(drift_confirmed) or (
        critical_window and next_critical_window_count >= confirm_windows
    )
    status = "confirmed" if drift_confirmed else "suspected" if drift_suspected else "clear"
    domain_score = _score_ratio(domain_ratio, domain_ratio_critical)
    degradation_score = max(
        _score_ratio(alert_rate, alert_rate_critical),
        _score_ratio(red_rate, red_rate_critical),
        _score_ratio(unexpected_red_rate, unexpected_red_rate_critical),
        _score_ratio(roi_fail_rate, roi_fail_rate_critical),
        _score_ratio(oracle_fn_rate, oracle_fn_rate_critical),
    )
    drift_score = domain_score * degradation_score
    return {
        "scenario_id": scenario_id,
        "status": status,
        "drift_score": drift_score,
        "drift_suspected": drift_suspected,
        "drift_confirmed": drift_confirmed,
        "critical_window": critical_window,
        "critical_window_count": next_critical_window_count,
        "confirm_windows": confirm_windows,
        "min_window_events": min_window_events,
        "signals": signals,
        "degradation_signals": degradation_signals,
        "metrics": {
            "window_events": window_events,
            "domain_ratio": domain_ratio,
            "alert_rate": alert_rate,
            "red_rate": red_rate,
            "unexpected_red_rate": unexpected_red_rate,
            "roi_fail_rate": roi_fail_rate,
            "oracle_fn_rate": oracle_fn_rate,
            "domain_score": domain_score,
            "degradation_score": degradation_score,
            "drift_score": drift_score,
        },
        "thresholds": drift_thresholds,
    }


def _evaluate_drift(args: argparse.Namespace, thresholds: dict[str, object]) -> dict[str, object]:
    return evaluate_drift_metrics(
        scenario_id=args.scenario_id,
        window_events=args.window_events,
        domain_ratio=args.domain_ratio,
        alert_rate=args.alert_rate,
        red_rate=args.red_rate,
        unexpected_red_rate=args.unexpected_red_rate,
        roi_fail_rate=args.roi_fail_rate,
        oracle_fn_rate=args.oracle_fn_rate,
        critical_window_count=args.critical_window_count,
        drift_confirmed=bool(args.drift_confirmed),
        thresholds=thresholds,
    )


def _push_drift_event(args: argparse.Namespace, drift_evaluation: dict[str, object]) -> dict[str, object]:
    api_url = str(args.api_url or "").strip()
    if not api_url:
        return {"attempted": False, "status": "skipped", "reason": "IQA_API_URL not configured"}
    endpoint = f"{api_url.rstrip('/')}/internal/drift/events"
    payload = {
        "event_type": "window_evaluated",
        "scenario_id": args.scenario_id,
        "status": drift_evaluation["status"],
        "source_domain": args.source_domain,
        "window_events": args.window_events,
        "trigger_lifecycle": bool(drift_evaluation.get("drift_confirmed")),
        "metrics": drift_evaluation["metrics"],
    }
    active_models = getattr(args, "active_models", None)
    if active_models:
        payload["active_models"] = active_models
    headers = {"Content-Type": "application/json"}
    if args.service_token:
        headers["X-IQA-Service-Token"] = args.service_token
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=2) as response:
            return {"attempted": True, "status": "sent", "http_status": response.status}
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {"attempted": True, "status": "warning", "reason": str(exc)}


def main() -> None:
    args = parse_args()
    thresholds = load_yaml_config(args.thresholds_config)
    drift_eval = _evaluate_drift(args, thresholds)
    signal = LifecycleSignal(
        scenario_id=args.scenario_id,
        conforming_validated_count=args.conforming_validated_count,
        drift_confirmed=bool(drift_eval["drift_confirmed"]),
        roi_fail_rate=args.roi_fail_rate,
    )
    decision = evaluate_lifecycle_signal(signal)
    roi_eval = _evaluate_roi_fail_rate(args.roi_fail_rate, thresholds)
    api_push = _push_drift_event(args, drift_eval)
    result = {
        "service": "iqa-monitoring",
        "signal": signal.to_dict(),
        "lifecycle_decision": decision.to_dict(),
        "trigger_lifecycle": should_trigger_lifecycle(signal),
        "trigger_reason": decision.trigger_reason,
        "drift_suspected": drift_eval["drift_suspected"],
        "drift_confirmed": drift_eval["drift_confirmed"],
        "drift_evaluation": drift_eval,
        "api_drift_push": api_push,
        # Real in-container threshold evaluation (configs/monitoring_thresholds.yaml).
        "thresholds_evaluated": bool(thresholds),
        "roi_fail_rate_evaluation": roi_eval,
        "status": "validated",
    }
    print_json(result)


if __name__ == "__main__":
    main()
