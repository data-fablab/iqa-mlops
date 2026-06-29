"""Backfill Grafana/Prometheus observability from local IQA run artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from scripts.airflow_contracts import print_json


LIFECYCLE_SCENARIO_DEFAULT = "production_replay_natural_piece_b_full"
DRIFT_SCENARIO_DEFAULT = "production_replay_natural_piece_b_to_piece_a_p4_drift"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-run-dir", type=Path)
    parser.add_argument("--drift-observation-dir", type=Path)
    parser.add_argument("--api-url", default=os.getenv("IQA_API_URL", "http://localhost:8002"))
    parser.add_argument("--service-token", default=os.getenv("IQA_SERVICE_TOKEN", ""))
    parser.add_argument("--pace-seconds", type=float, default=6.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events: list[dict[str, Any]] = []
    if args.lifecycle_run_dir:
        events.extend(build_lifecycle_events(args.lifecycle_run_dir))
    if args.drift_observation_dir:
        events.extend(build_drift_events(args.drift_observation_dir))
    result = replay_events(
        events,
        api_url=args.api_url,
        service_token=args.service_token,
        pace_seconds=args.pace_seconds,
        progress_every=0 if args.quiet else args.progress_every,
        dry_run=args.dry_run,
    )
    print_json(result)


def build_lifecycle_events(run_dir: Path) -> list[dict[str, Any]]:
    progress = _read_json(run_dir / "progress.json")
    summary = _read_json(run_dir / "summary.json")
    cycles = _read_jsonl(run_dir / "cycles.jsonl")
    scenario_id = str(progress.get("scenario_id") or summary.get("scenario_id") or LIFECYCLE_SCENARIO_DEFAULT)
    lifecycle_run_id = str(progress.get("run_id") or summary.get("run_id") or run_dir.name)
    events: list[dict[str, Any]] = []

    for cycle in cycles:
        cycle_id = str(cycle.get("cycle_id") or "cycle_000")
        candidate_version = str(cycle.get("candidate_version") or "")
        candidate_init_policy = str(cycle.get("candidate_init_policy") or progress.get("candidate_init_policy") or "")
        events.extend(
            _epoch_events(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                candidate_init_policy=candidate_init_policy,
            )
        )
        events.append(
            _promotion_event(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                candidate_init_policy=candidate_init_policy,
            )
        )

    events.append(_run_completed_event(progress, summary, scenario_id=scenario_id, lifecycle_run_id=lifecycle_run_id))
    return events


def build_drift_events(observation_dir: Path) -> list[dict[str, Any]]:
    summary = _read_json(observation_dir / "summary.json")
    windows = _read_jsonl(observation_dir / "windows.jsonl")
    scenario_id = str(summary.get("scenario_id") or DRIFT_SCENARIO_DEFAULT)
    first_confirmed = summary.get("first_confirmed_window_index")
    active_models = _drift_active_models(summary)
    events: list[dict[str, Any]] = []
    for window in windows:
        metrics = dict(window.get("metrics") or {})
        window_index = _int_or_none(window.get("window_index"))
        payload = {
            "event_type": "window_evaluated",
            "scenario_id": scenario_id,
            "status": str(window.get("status") or "clear"),
            "source_domain": "piece_a_p4",
            "window_index": window_index,
            "first_confirmed_window_index": _int_or_none(first_confirmed),
            "window_events": _int_or_none(metrics.get("window_events")),
            "trigger_lifecycle": bool(window.get("drift_confirmed")),
            "active_models": active_models,
            "metrics": _finite_metrics(metrics),
        }
        events.append(payload)
    return events


def replay_events(
    events: list[dict[str, Any]],
    *,
    api_url: str,
    service_token: str = "",
    pace_seconds: float = 0.0,
    progress_every: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    sent = 0
    skipped = 0
    failures: list[str] = []
    total = len(events)
    started_at = time.monotonic()
    if progress_every > 0:
        sleep_count = max(total - 1, 0) if not dry_run and pace_seconds > 0 else 0
        estimated_seconds = sleep_count * max(pace_seconds, 0)
        _print_progress(
            f"Backfill observability: {total} events, pace={pace_seconds:g}s, "
            f"estimated_duration={_format_duration(estimated_seconds)}"
        )
    for index, event in enumerate(events, start=1):
        status = "skipped"
        if dry_run:
            skipped += 1
        else:
            try:
                _post_event(event, api_url=api_url, service_token=service_token)
                sent += 1
                status = "sent"
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                failures.append(f"{event.get('event_type')}:{type(exc).__name__}:{exc}")
                status = f"failed:{type(exc).__name__}"
        if progress_every > 0 and (index == 1 or index == total or index % progress_every == 0):
            elapsed = time.monotonic() - started_at
            _print_progress(
                f"[{index}/{total}] {status} {_event_summary(event)} "
                f"elapsed={_format_duration(elapsed)}"
            )
        if not dry_run and pace_seconds > 0 and index < total:
            time.sleep(pace_seconds)
    return {
        "api_url": api_url,
        "dry_run": dry_run,
        "events_total": len(events),
        "events_sent": sent,
        "events_skipped": skipped,
        "failures": failures,
        "status": "validated" if not failures else "warning",
    }


def _epoch_events(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    candidate_init_policy: str,
) -> list[dict[str, Any]]:
    run_dir_value = cycle.get("candidate_run_dir")
    if not run_dir_value:
        return []
    epoch_path = Path(str(run_dir_value)) / "epoch_metrics.jsonl"
    if not epoch_path.is_file():
        return []
    events = []
    for record in _read_jsonl(epoch_path):
        metrics = record.get("metrics") or {}
        payload_metrics = {
            "pixel_aupimo": metrics.get("pixel_aupimo_1e-5_1e-3"),
            "pixel_ap": metrics.get("pixel_ap"),
            "image_ap": metrics.get("image_ap"),
            "false_negatives": metrics.get("false_negatives"),
        }
        events.append(
            {
                "event_type": "epoch_completed",
                "scenario_id": scenario_id,
                "lifecycle_run_id": lifecycle_run_id,
                "cycle_id": cycle_id,
                "epoch": _int_or_none(record.get("epoch")),
                "candidate_version": candidate_version,
                "candidate_init_policy": candidate_init_policy,
                "metrics": _finite_metrics(payload_metrics),
            }
        )
    return events


def _promotion_event(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    candidate_init_policy: str,
) -> dict[str, Any]:
    metrics = {
        "gate_localization_active_pixel_aupimo": _metric(cycle, "localization_active_metrics_on_eval_set", "pixel_aupimo_1e-5_1e-3"),
        "gate_localization_candidate_pixel_aupimo": _metric(cycle, "localization_candidate_metrics_on_eval_set", "pixel_aupimo_1e-5_1e-3"),
        "gate_delta_localization_pixel_aupimo": cycle.get("localization_metric_delta"),
        "gate_localization_active_pixel_ap": _metric(cycle, "localization_active_metrics_on_eval_set", "pixel_ap"),
        "gate_localization_candidate_pixel_ap": _metric(cycle, "localization_candidate_metrics_on_eval_set", "pixel_ap"),
        "gate_delta_localization_pixel_ap": _delta(
            _metric(cycle, "localization_candidate_metrics_on_eval_set", "pixel_ap"),
            _metric(cycle, "localization_active_metrics_on_eval_set", "pixel_ap"),
        ),
        "gate_classification_active_false_negatives": _metric(cycle, "classification_active_metrics_on_eval_set", "false_negatives"),
        "gate_classification_candidate_false_negatives": _metric(cycle, "classification_candidate_metrics_on_eval_set", "false_negatives"),
        "gate_delta_classification_false_negatives": cycle.get("classification_metric_delta"),
        "gate_classification_active_image_ap": _metric(cycle, "classification_active_metrics_on_eval_set", "image_ap"),
        "gate_classification_candidate_image_ap": _metric(cycle, "classification_candidate_metrics_on_eval_set", "image_ap"),
        "gate_delta_classification_image_ap": _delta(
            _metric(cycle, "classification_candidate_metrics_on_eval_set", "image_ap"),
            _metric(cycle, "classification_active_metrics_on_eval_set", "image_ap"),
        ),
        "gate_classification_active_image_recall": _metric(cycle, "classification_active_metrics_on_eval_set", "image_recall"),
        "gate_classification_candidate_image_recall": _metric(cycle, "classification_candidate_metrics_on_eval_set", "image_recall"),
    }
    return {
        "event_type": "promotion_decision",
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id,
        "candidate_version": candidate_version,
        "candidate_init_policy": candidate_init_policy,
        "localization_promotion_status": cycle.get("localization_promotion_status"),
        "classification_promotion_status": cycle.get("classification_promotion_status"),
        "localization_gate_reason": cycle.get("localization_gate_reason"),
        "classification_gate_reason": cycle.get("classification_gate_reason"),
        "active_classification_model_version": _runtime_version(cycle.get("active_classification_runtime_after")),
        "active_localization_model_version": _runtime_version(cycle.get("active_localization_runtime_after")),
        "metrics": _finite_metrics(metrics),
    }


def _run_completed_event(
    progress: dict[str, Any],
    summary: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
) -> dict[str, Any]:
    classification_runtime = summary.get("active_classification_runtime_final") or {}
    localization_runtime = summary.get("active_localization_runtime_final") or {}
    return {
        "event_type": "run_completed",
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": _last_cycle_id(progress),
        "candidate_version": str(progress.get("active_classification_model_version") or ""),
        "active_classification_model_version": _runtime_version(classification_runtime),
        "active_classification_registered_model_name": classification_runtime.get("registry_model_name"),
        "active_classification_registered_model_version": classification_runtime.get("registered_model_version"),
        "active_localization_model_version": _runtime_version(localization_runtime),
        "active_localization_registered_model_name": localization_runtime.get("registry_model_name"),
        "active_localization_registered_model_version": localization_runtime.get("registered_model_version"),
        "metrics": _finite_metrics(
            {
                "events_processed": progress.get("events_processed") or summary.get("events_processed"),
                "cycles_completed": progress.get("cycles_completed") or summary.get("cycles_completed"),
            }
        ),
    }


def _post_event(event: dict[str, Any], *, api_url: str, service_token: str) -> None:
    event_type = str(event.get("event_type") or "")
    endpoint_name = "drift" if event_type == "window_evaluated" else "lifecycle"
    endpoint = f"{api_url.rstrip('/')}/internal/{endpoint_name}/events"
    headers = {"Content-Type": "application/json"}
    if service_token:
        headers["X-IQA-Service-Token"] = service_token
    request = Request(endpoint, data=json.dumps(event).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=5) as response:
        response.read()


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _event_summary(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "unknown")
    scenario_id = str(event.get("scenario_id") or "")
    cycle_id = str(event.get("cycle_id") or "")
    epoch = event.get("epoch")
    window_index = event.get("window_index")
    status = str(event.get("status") or "")
    parts = [event_type]
    if scenario_id:
        parts.append(f"scenario={scenario_id}")
    if cycle_id:
        parts.append(f"cycle={cycle_id}")
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if window_index is not None:
        parts.append(f"window={window_index}")
    if status:
        parts.append(f"status={status}")
    return " ".join(parts)


def _drift_active_models(summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for role, key in (
        ("classification", "active_classification_runtime"),
        ("localization", "active_localization_runtime"),
    ):
        runtime = summary.get(key) or {}
        result[role] = {
            "version": str(runtime.get("version") or ""),
            "registry_model_name": str(runtime.get("registry_model_name") or ""),
            "registered_model_version": str(runtime.get("registered_model_version") or ""),
            "registry_stage": str(runtime.get("registry_stage") or ""),
            "runtime_contract_status": str(runtime.get("runtime_contract_status") or ""),
        }
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        if value is None or isinstance(value, bool):
            if isinstance(value, bool):
                result[key] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and number not in {float("inf"), float("-inf")}:
            result[key] = int(number) if number.is_integer() else number
    return result


def _metric(cycle: dict[str, Any], section: str, metric_name: str) -> Any:
    metrics = cycle.get(section) or {}
    return metrics.get(metric_name)


def _delta(candidate: Any, active: Any) -> float | None:
    try:
        return float(candidate) - float(active)
    except (TypeError, ValueError):
        return None


def _runtime_version(runtime: Any) -> str:
    if isinstance(runtime, dict):
        return str(runtime.get("version") or "")
    return ""


def _last_cycle_id(progress: dict[str, Any]) -> str:
    cycles_completed = progress.get("cycles_completed")
    try:
        return f"cycle_{int(cycles_completed):03d}"
    except (TypeError, ValueError):
        return str((progress.get("last_cycle") or {}).get("cycle_id") or "cycle_000")


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
