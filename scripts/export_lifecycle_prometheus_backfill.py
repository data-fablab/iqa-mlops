"""Export historical lifecycle observability as Prometheus OpenMetrics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_LIFECYCLE_ROOT = Path(".cache/iqa/replay_lifecycle")
DEFAULT_OUTPUT = Path(".cache/iqa/prometheus_backfill/lifecycle_selected_epochs.prom")
BACKFILL_CADENCE_MS = 30_000
STATEFUL_METRICS = {
    "iqa_lifecycle_active_model_info",
    "iqa_lifecycle_final_model_info",
    "iqa_lifecycle_gate_delta",
    "iqa_lifecycle_gate_fn_delta",
    "iqa_lifecycle_gate_metric_delta",
    "iqa_lifecycle_gate_value",
    "iqa_lifecycle_promotion_decision_info",
    "iqa_lifecycle_promotion_selected_epoch",
    "iqa_lifecycle_promotion_selected_metric_value",
    "iqa_lifecycle_promotion_total",
    "iqa_lifecycle_run_cycles_completed",
    "iqa_lifecycle_run_events_processed",
    "iqa_lifecycle_train_set_size",
}
EPOCH_TRACE_METRICS = {
    "iqa_lifecycle_cycle_current",
    "iqa_lifecycle_epoch_current",
    "iqa_lifecycle_epoch_image_ap",
    "iqa_lifecycle_epoch_metric",
    "iqa_lifecycle_epoch_pixel_ap",
    "iqa_lifecycle_epoch_pixel_aupimo",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = sorted(
        {path.parent for path in args.lifecycle_root.rglob("progress.json")},
        key=lambda path: path.stat().st_mtime,
    )
    samples = []
    for run_dir in run_dirs:
        samples.extend(_promotion_selection_samples(run_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(_openmetrics(samples))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runs": len(run_dirs),
                "samples": len(samples),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _promotion_selection_samples(run_dir: Path) -> list[dict[str, Any]]:
    progress = _read_json(run_dir / "progress.json")
    summary = _read_json(run_dir / "summary.json")
    cycles = _read_jsonl(run_dir / "cycles.jsonl")
    event_timestamps = _promotion_event_timestamps(run_dir / "lifecycle_events.jsonl")
    train_windows = _train_windows(run_dir / "timings.jsonl")
    scenario_id = str(progress.get("scenario_id") or summary.get("scenario_id") or "")
    lifecycle_run_id = str(progress.get("run_id") or summary.get("run_id") or run_dir.name)
    samples: list[dict[str, Any]] = []
    promotion_counter_values: dict[tuple[tuple[str, str], ...], int] = {}
    promotion_counter_initialized: set[tuple[tuple[str, str], ...]] = set()

    for cycle in cycles:
        cycle_id = str(cycle.get("cycle_id") or "cycle_000")
        candidate_version = str(cycle.get("candidate_version") or "")
        candidate_init_policy = str(cycle.get("candidate_init_policy") or "")
        samples.extend(
            _epoch_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                candidate_init_policy=candidate_init_policy,
                train_window=train_windows.get(cycle_id),
            )
        )
        timestamp = (
            event_timestamps.get(cycle_id)
            or _parse_timestamp(cycle.get("evaluation_completed_at"))
            or _parse_timestamp(cycle.get("created_at"))
        )
        if timestamp is None:
            continue
        samples.extend(
            _training_set_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                candidate_init_policy=candidate_init_policy,
                timestamp_ms=timestamp,
            )
        )
        samples.extend(
            _gate_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                timestamp_ms=timestamp,
            )
        )
        samples.extend(
            _piece_b_report_value_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                timestamp_ms=timestamp,
            )
        )
        samples.extend(
            _promotion_decision_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                candidate_version=candidate_version,
                timestamp_ms=timestamp,
            )
        )
        for sample in _role_samples(
            cycle,
            scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                timestamp_ms=timestamp,
            ):
            samples.append(sample)
        samples.extend(
            _active_model_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                cycle_id=cycle_id,
                timestamp_ms=timestamp,
            )
        )
        samples.extend(
            _promotion_counter_samples(
                cycle,
                scenario_id=scenario_id,
                lifecycle_run_id=lifecycle_run_id,
                timestamp_ms=timestamp,
                values=promotion_counter_values,
                initialized=promotion_counter_initialized,
            )
        )
    samples.extend(
        _run_summary_samples(
            progress,
            cycles,
            scenario_id=scenario_id,
            lifecycle_run_id=lifecycle_run_id,
        )
    )
    run_end_ms = _run_end_timestamp(progress, cycles)
    return _scrape_like_samples(samples, run_end_ms=run_end_ms)


def _epoch_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    candidate_init_policy: str,
    train_window: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    history = cycle.get("epoch_metric_history") or []
    if not isinstance(history, list) or not history:
        return []
    output: list[dict[str, Any]] = []
    total_epochs = len(history)
    fallback_end = _parse_timestamp(cycle.get("evaluation_completed_at")) or _parse_timestamp(cycle.get("created_at"))
    if train_window is not None:
        start_ms, end_ms = train_window
    elif fallback_end is not None:
        end_ms = fallback_end
        start_ms = end_ms - total_epochs * 60_000
    else:
        return []
    base_labels = {
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id,
        "candidate_version": candidate_version,
        "candidate_init_policy": candidate_init_policy,
    }
    cycle_number = _cycle_number(cycle_id)
    for index, record in enumerate(history, start=1):
        if not isinstance(record, dict):
            continue
        epoch = _int_or_none(record.get("epoch")) or index
        timestamp_ms = start_ms + int((end_ms - start_ms) * index / max(total_epochs, 1))
        metrics = record.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        output.append(_sample("iqa_lifecycle_cycle_current", base_labels, cycle_number, timestamp_ms))
        output.append(_sample("iqa_lifecycle_epoch_current", base_labels, epoch, timestamp_ms))
        metric_specs = {
            "pixel_aupimo_1e-5_1e-3": ("pixel_aupimo", "iqa_lifecycle_epoch_pixel_aupimo"),
            "pixel_aupimo": ("pixel_aupimo", "iqa_lifecycle_epoch_pixel_aupimo"),
            "pixel_ap": ("pixel_ap", "iqa_lifecycle_epoch_pixel_ap"),
            "image_ap": ("image_ap", "iqa_lifecycle_epoch_image_ap"),
            "false_negatives": ("false_negatives", None),
        }
        for source_name, (metric_label, dedicated_metric) in metric_specs.items():
            value = _finite_float(metrics.get(source_name))
            if value is None:
                continue
            metric_labels = dict(base_labels)
            metric_labels["metric"] = metric_label
            output.append(_sample("iqa_lifecycle_epoch_metric", metric_labels, value, timestamp_ms))
            if dedicated_metric is not None:
                output.append(_sample(dedicated_metric, base_labels, value, timestamp_ms))
    return output


def _role_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specs = (
        (
            "localization",
            cycle.get("localization_promotion_status"),
            _int_or_none(cycle.get("localization_selected_epoch") or cycle.get("selected_epoch") or cycle.get("epoch_selected_epoch")),
            cycle.get("localization_selected_metric") or cycle.get("selected_metric") or cycle.get("epoch_selected_metric"),
            _finite_float(cycle.get("localization_selected_metric_value") or cycle.get("localization_candidate_metric_value") or cycle.get("selected_metric_value")),
        ),
        (
            "classification",
            cycle.get("classification_promotion_status"),
            _classification_selected_epoch(cycle),
            cycle.get("classification_selected_metric"),
            _finite_float(cycle.get("classification_selected_metric_value") or cycle.get("classification_candidate_metric_value")),
        ),
    )
    for role, status, selected_epoch, selected_metric, selected_metric_value in specs:
        if status != "promoted" or selected_epoch is None:
            continue
        labels = {
            "scenario_id": scenario_id,
            "lifecycle_run_id": lifecycle_run_id,
            "cycle_id": cycle_id,
            "role": role,
            "status": str(status),
            "candidate_version": str(cycle.get("candidate_version") or ""),
            "selected_metric": str(selected_metric or ""),
            "mlflow_run_id": str(cycle.get("mlflow_run_id") or ""),
        }
        output.append(
            {
                "metric": "iqa_lifecycle_promotion_selected_epoch",
                "labels": labels,
                "value": selected_epoch,
                "timestamp_ms": timestamp_ms,
            }
        )
        if selected_metric_value is not None:
            output.append(
                {
                    "metric": "iqa_lifecycle_promotion_selected_metric_value",
                    "labels": labels,
                    "value": selected_metric_value,
                    "timestamp_ms": timestamp_ms,
                }
            )
    return output


def _gate_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    base_labels = {
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id,
        "candidate_version": candidate_version,
    }
    for role, delta_key in (
        ("localization", "localization_metric_delta"),
        ("classification", "classification_metric_delta"),
    ):
        metric_delta = _finite_float(cycle.get(delta_key))
        if metric_delta is not None:
            labels = dict(base_labels)
            labels["role"] = role
            output.append(_sample("iqa_lifecycle_gate_metric_delta", labels, metric_delta, timestamp_ms))
    fn_delta = _finite_float(cycle.get("classification_fn_delta") or cycle.get("fn_delta"))
    if fn_delta is not None:
        labels = dict(base_labels)
        labels["role"] = "classification"
        output.append(_sample("iqa_lifecycle_gate_fn_delta", labels, fn_delta, timestamp_ms))
    output.extend(
        _role_gate_value_samples(
            cycle,
            role="localization",
            gate=cycle.get("localization_gate"),
            base_labels=base_labels,
            timestamp_ms=timestamp_ms,
        )
    )
    output.extend(
        _classification_gate_value_samples(
            cycle,
            gate=cycle.get("classification_gate"),
            base_labels=base_labels,
            timestamp_ms=timestamp_ms,
        )
    )
    return output


def _role_gate_value_samples(
    cycle: dict[str, Any],
    *,
    role: str,
    gate: Any,
    base_labels: dict[str, str],
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    if not isinstance(gate, dict):
        return []
    metric_name = str(gate.get("metric") or cycle.get(f"{role}_selected_metric") or "pixel_aupimo")
    output: list[dict[str, Any]] = []
    for model, source_key in (("active", "active_value"), ("candidate", "candidate_value")):
        value = _finite_float(gate.get(source_key))
        if value is None:
            continue
        labels = dict(base_labels)
        labels.update({"role": role, "model": model, "metric": _metric_label(metric_name)})
        output.append(_sample("iqa_lifecycle_gate_value", labels, value, timestamp_ms))
    output.extend(
        _metric_pair_samples(
            cycle,
            base_labels=base_labels,
            role=role,
            metrics=("pixel_aupimo_1e-5_1e-3", "pixel_ap"),
            active_key=f"{role}_active_metrics_on_eval_set",
            candidate_key=f"{role}_candidate_metrics_on_eval_set",
            timestamp_ms=timestamp_ms,
        )
    )
    delta = _finite_float(gate.get("delta") or cycle.get(f"{role}_metric_delta"))
    if delta is not None:
        labels = dict(base_labels)
        labels.update({"role": role, "metric": _metric_label(metric_name)})
        output.append(_sample("iqa_lifecycle_gate_delta", labels, delta, timestamp_ms))
    return output


def _classification_gate_value_samples(
    cycle: dict[str, Any],
    *,
    gate: Any,
    base_labels: dict[str, str],
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    if not isinstance(gate, dict):
        return []
    output: list[dict[str, Any]] = []
    metric_specs = (
        ("false_negatives", "active_false_negatives", "candidate_false_negatives", "fn_delta"),
        ("image_ap", "active_image_ap", "candidate_image_ap", "image_ap_delta"),
        ("image_recall", "active_image_recall", "candidate_image_recall", "image_recall_delta"),
        ("good_red_count", "active_good_red_count", "candidate_good_red_count", "good_red_delta"),
    )
    for metric_name, active_key, candidate_key, delta_key in metric_specs:
        for model, source_key in (("active", active_key), ("candidate", candidate_key)):
            value = _finite_float(gate.get(source_key))
            if value is None:
                continue
            labels = dict(base_labels)
            labels.update({"role": "classification", "model": model, "metric": metric_name})
            output.append(_sample("iqa_lifecycle_gate_value", labels, value, timestamp_ms))
        delta = _finite_float(gate.get(delta_key))
        if delta is not None:
            labels = dict(base_labels)
            labels.update({"role": "classification", "metric": metric_name})
            output.append(_sample("iqa_lifecycle_gate_delta", labels, delta, timestamp_ms))
    return output


def _metric_pair_samples(
    cycle: dict[str, Any],
    *,
    base_labels: dict[str, str],
    role: str,
    metrics: tuple[str, ...],
    active_key: str,
    candidate_key: str,
    timestamp_ms: int,
    metric_prefix: str = "",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model, source_key in (("active", active_key), ("candidate", candidate_key)):
        payload = cycle.get(source_key)
        if not isinstance(payload, dict):
            continue
        for metric_name in metrics:
            value = _finite_float(payload.get(metric_name))
            if value is None:
                continue
            labels = dict(base_labels)
            labels.update(
                {
                    "role": role,
                    "model": model,
                    "metric": f"{metric_prefix}{_metric_label(metric_name)}",
                }
            )
            output.append(_sample("iqa_lifecycle_gate_value", labels, value, timestamp_ms))
    return output


def _piece_b_report_value_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    base_labels = {
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id,
        "candidate_version": candidate_version,
    }
    output: list[dict[str, Any]] = []
    for role in ("classification", "localization"):
        output.extend(
            _metric_pair_samples(
                cycle,
                base_labels=base_labels,
                role=role,
                metrics=("false_negatives", "image_ap", "image_recall", "pixel_aupimo_1e-5_1e-3", "pixel_ap"),
                active_key=f"piece_b_non_regression_{role}_active_metrics",
                candidate_key=f"piece_b_non_regression_{role}_candidate_metrics",
                timestamp_ms=timestamp_ms,
                metric_prefix="piece_b_",
            )
        )
    return output


def _promotion_decision_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role, status in (
        ("localization", cycle.get("localization_promotion_status")),
        ("classification", cycle.get("classification_promotion_status")),
    ):
        if not status:
            continue
        labels = {
            "scenario_id": scenario_id,
            "lifecycle_run_id": lifecycle_run_id,
            "cycle_id": cycle_id,
            "role": role,
            "status": str(status),
            "candidate_version": candidate_version,
            "mlflow_run_id": str(cycle.get("mlflow_run_id") or ""),
        }
        output.append(_sample("iqa_lifecycle_promotion_decision_info", labels, 1, timestamp_ms))
    return output


def _active_model_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role, runtime_key in (
        ("classification", "active_classification_runtime_after"),
        ("localization", "active_localization_runtime_after"),
    ):
        runtime = cycle.get(runtime_key) or {}
        if not isinstance(runtime, dict):
            runtime = {}
        version = str(runtime.get("version") or cycle.get("candidate_version") or "")
        if not version:
            continue
        labels = {
            "scenario_id": scenario_id,
            "role": role,
            "version": version,
            "run_id": lifecycle_run_id,
            "cycle_id": cycle_id,
        }
        output.append(_sample("iqa_lifecycle_active_model_info", labels, 1, timestamp_ms))
    return output


def _final_model_samples(
    progress: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role, runtime_key, version_key in (
        ("classification", "active_classification_runtime_model", "active_classification_model_version"),
        ("localization", "active_localization_runtime_model", "active_localization_model_version"),
    ):
        runtime = progress.get(runtime_key) or {}
        if not isinstance(runtime, dict):
            runtime = {}
        version = str(runtime.get("version") or progress.get(version_key) or "")
        if not version:
            continue
        labels = {
            "scenario_id": scenario_id,
            "lifecycle_run_id": lifecycle_run_id,
            "role": role,
            "version": version,
            "registered_model_version": str(runtime.get("registered_model_version") or ""),
            "registered_model_name": str(runtime.get("registry_model_name") or ""),
        }
        output.append(_sample("iqa_lifecycle_final_model_info", labels, 1, timestamp_ms))
    return output


def _run_summary_samples(
    progress: dict[str, Any],
    cycles: list[dict[str, Any]],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
) -> list[dict[str, Any]]:
    timestamp = _parse_timestamp(progress.get("updated_at"))
    if timestamp is None and cycles:
        timestamp = (
            _parse_timestamp(cycles[-1].get("evaluation_completed_at"))
            or _parse_timestamp(cycles[-1].get("created_at"))
        )
    if timestamp is None:
        return []
    cycle_id = str(progress.get("last_cycle", {}).get("cycle_id") if isinstance(progress.get("last_cycle"), dict) else "")
    if not cycle_id and cycles:
        cycle_id = str(cycles[-1].get("cycle_id") or "cycle_000")
    candidate_version = str(progress.get("last_cycle", {}).get("candidate_version") if isinstance(progress.get("last_cycle"), dict) else "")
    if not candidate_version and cycles:
        candidate_version = str(cycles[-1].get("candidate_version") or "")
    base_labels = {
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id or "cycle_000",
        "candidate_version": candidate_version,
        "candidate_init_policy": str(cycles[-1].get("candidate_init_policy") or "") if cycles else "",
    }
    output: list[dict[str, Any]] = []
    for source_key, metric_name in (
        ("events_processed", "iqa_lifecycle_run_events_processed"),
        ("cycles_completed", "iqa_lifecycle_run_cycles_completed"),
    ):
        value = _finite_float(progress.get(source_key))
        if value is not None:
            output.append(_sample(metric_name, base_labels, value, timestamp))
    output.extend(
        _final_model_samples(
            progress,
            scenario_id=scenario_id,
            lifecycle_run_id=lifecycle_run_id,
            timestamp_ms=timestamp,
        )
    )
    return output


def _training_set_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    cycle_id: str,
    candidate_version: str,
    candidate_init_policy: str,
    timestamp_ms: int,
) -> list[dict[str, Any]]:
    stats = cycle.get("training_manifest_stats")
    if not isinstance(stats, dict):
        return []

    output: list[dict[str, Any]] = []
    base_labels = {
        "scenario_id": scenario_id,
        "lifecycle_run_id": lifecycle_run_id,
        "cycle_id": cycle_id,
        "candidate_version": candidate_version,
        "candidate_init_policy": candidate_init_policy,
    }
    for kind, source_key in (
        ("total", "total_count"),
        ("seen_conforming", "seen_conforming_count"),
        ("anchor_good", "anchor_good_count"),
    ):
        value = _finite_float(stats.get(source_key))
        if value is None:
            continue
        labels = dict(base_labels)
        labels["kind"] = kind
        output.append(_sample("iqa_lifecycle_train_set_size", labels, value, timestamp_ms))
    return output


def _run_end_timestamp(progress: dict[str, Any], cycles: list[dict[str, Any]]) -> int | None:
    candidates = [_parse_timestamp(progress.get("updated_at"))]
    for cycle in cycles:
        candidates.extend(
            [
                _parse_timestamp(cycle.get("evaluation_completed_at")),
                _parse_timestamp(cycle.get("created_at")),
            ]
        )
    return max((value for value in candidates if value is not None), default=None)


def _scrape_like_samples(samples: list[dict[str, Any]], *, run_end_ms: int | None) -> list[dict[str, Any]]:
    if run_end_ms is None:
        return samples
    grouped: dict[tuple[str, tuple[tuple[str, str], ...]], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for sample in samples:
        metric = str(sample["metric"])
        if metric not in STATEFUL_METRICS and metric not in EPOCH_TRACE_METRICS:
            passthrough.append(sample)
            continue
        key = (metric, tuple(sorted(sample["labels"].items())))
        grouped.setdefault(key, []).append(sample)

    output = list(passthrough)
    for (metric, _labels_key), series_samples in grouped.items():
        ordered = sorted(series_samples, key=lambda item: item["timestamp_ms"])
        if metric in STATEFUL_METRICS:
            output.extend(_hold_series_until(ordered, end_ms=run_end_ms))
        else:
            output.extend(_fill_series_between_samples(ordered))
    return output


def _hold_series_until(samples: list[dict[str, Any]], *, end_ms: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        start_ms = int(sample["timestamp_ms"])
        next_ms = int(samples[index + 1]["timestamp_ms"]) if index + 1 < len(samples) else end_ms
        output.append(sample)
        if next_ms <= start_ms:
            continue
        timestamp_ms = _next_cadence(start_ms)
        while timestamp_ms < next_ms:
            output.append(_sample(sample["metric"], dict(sample["labels"]), sample["value"], timestamp_ms))
            timestamp_ms += BACKFILL_CADENCE_MS
    last_sample = samples[-1]
    if int(last_sample["timestamp_ms"]) < end_ms:
        output.append(_sample(last_sample["metric"], dict(last_sample["labels"]), last_sample["value"], end_ms))
    return output


def _fill_series_between_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        output.append(sample)
        if index + 1 >= len(samples):
            continue
        start_ms = int(sample["timestamp_ms"])
        next_ms = int(samples[index + 1]["timestamp_ms"])
        timestamp_ms = _next_cadence(start_ms)
        while timestamp_ms < next_ms:
            output.append(_sample(sample["metric"], dict(sample["labels"]), sample["value"], timestamp_ms))
            timestamp_ms += BACKFILL_CADENCE_MS
    return output


def _next_cadence(timestamp_ms: int) -> int:
    return ((timestamp_ms // BACKFILL_CADENCE_MS) + 1) * BACKFILL_CADENCE_MS


def _promotion_counter_samples(
    cycle: dict[str, Any],
    *,
    scenario_id: str,
    lifecycle_run_id: str,
    timestamp_ms: int,
    values: dict[tuple[tuple[str, str], ...], int],
    initialized: set[tuple[tuple[str, str], ...]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role, status in (
        ("localization", cycle.get("localization_promotion_status")),
        ("classification", cycle.get("classification_promotion_status")),
    ):
        if not status:
            continue
        labels = {
            "scenario_id": scenario_id,
            "lifecycle_run_id": lifecycle_run_id,
            "role": role,
            "status": str(status),
        }
        key = tuple(sorted(labels.items()))
        if key not in initialized:
            output.append(_sample("iqa_lifecycle_promotion_total", labels, 0, timestamp_ms - 1000))
            initialized.add(key)
        values[key] = values.get(key, 0) + 1
        output.append(_sample("iqa_lifecycle_promotion_total", labels, values[key], timestamp_ms))
    return output


def _promotion_event_timestamps(path: Path) -> dict[str, int]:
    timestamps: dict[str, int] = {}
    priority = {
        "gate_decision": 1,
        "dual_gate_decision": 2,
        "model_activated": 3,
    }
    chosen_priority: dict[str, int] = {}
    for event in _read_jsonl(path):
        cycle_id = event.get("cycle_id")
        event_type = str(event.get("event_type") or "")
        timestamp = _parse_timestamp(event.get("timestamp"))
        if not cycle_id or timestamp is None or event_type not in priority:
            continue
        current_priority = chosen_priority.get(str(cycle_id), 0)
        if priority[event_type] >= current_priority:
            timestamps[str(cycle_id)] = timestamp
            chosen_priority[str(cycle_id)] = priority[event_type]
    return timestamps


def _train_windows(path: Path) -> dict[str, tuple[int, int]]:
    windows: dict[str, tuple[int, int]] = {}
    for row in _read_jsonl(path):
        if row.get("phase") != "train":
            continue
        cycle_id = row.get("cycle_id")
        end_ms = _parse_timestamp(row.get("timestamp"))
        duration = _finite_float(row.get("duration_seconds"))
        if not cycle_id or end_ms is None or duration is None:
            continue
        start_ms = end_ms - int(duration * 1000)
        windows[str(cycle_id)] = (start_ms, end_ms)
    return windows


def _classification_selected_epoch(cycle: dict[str, Any]) -> int | None:
    explicit_epoch = _int_or_none(cycle.get("classification_selected_epoch"))
    if explicit_epoch is not None:
        return explicit_epoch
    checkpoint = Path(str(cycle.get("classification_candidate_checkpoint") or ""))
    if checkpoint.name.startswith("checkpoint_epoch_"):
        return _int_or_none(checkpoint.stem.removeprefix("checkpoint_epoch_"))
    metric_by_checkpoint = {
        "checkpoint_best_image.pt": "image_ap",
        "checkpoint_best_image_ap.pt": "image_ap",
        "checkpoint_best_image_auroc.pt": "image_auroc",
    }
    metric = metric_by_checkpoint.get(checkpoint.name)
    if metric is None:
        return None
    candidate_run_dir = cycle.get("candidate_run_dir")
    metric_eval_best = Path(str(candidate_run_dir)) / "metric_eval_best.json" if candidate_run_dir else checkpoint.parent / "metric_eval_best.json"
    payload = _read_json(metric_eval_best)
    record = payload.get(metric)
    if not isinstance(record, dict):
        return None
    return _int_or_none(record.get("epoch"))


def _openmetrics(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# TYPE iqa_lifecycle_active_model_info gauge",
        "# TYPE iqa_lifecycle_cycle_current gauge",
        "# TYPE iqa_lifecycle_epoch_current gauge",
        "# TYPE iqa_lifecycle_epoch_image_ap gauge",
        "# TYPE iqa_lifecycle_epoch_metric gauge",
        "# TYPE iqa_lifecycle_epoch_pixel_ap gauge",
        "# TYPE iqa_lifecycle_epoch_pixel_aupimo gauge",
        "# TYPE iqa_lifecycle_final_model_info gauge",
        "# TYPE iqa_lifecycle_gate_delta gauge",
        "# TYPE iqa_lifecycle_gate_fn_delta gauge",
        "# TYPE iqa_lifecycle_gate_metric_delta gauge",
        "# TYPE iqa_lifecycle_gate_value gauge",
        "# TYPE iqa_lifecycle_promotion_decision_info gauge",
        "# TYPE iqa_lifecycle_promotion_selected_epoch gauge",
        "# TYPE iqa_lifecycle_promotion_selected_metric_value gauge",
        "# TYPE iqa_lifecycle_promotion_total counter",
        "# TYPE iqa_lifecycle_run_cycles_completed gauge",
        "# TYPE iqa_lifecycle_run_events_processed gauge",
        "# TYPE iqa_lifecycle_train_set_size gauge",
    ]
    for sample in sorted(samples, key=lambda item: (item["metric"], item["timestamp_ms"], str(item["labels"]))):
        lines.append(
            f"{sample['metric']}{{{_labels(sample['labels'])}}} "
            f"{_number(sample['value'])} {_openmetrics_timestamp(sample['timestamp_ms'])}"
        )
    lines.append("# EOF")
    return "\n".join(lines) + "\n"


def _sample(metric: str, labels: dict[str, str], value: float | int, timestamp_ms: int) -> dict[str, Any]:
    return {
        "metric": metric,
        "labels": labels,
        "value": value,
        "timestamp_ms": timestamp_ms,
    }


def _cycle_number(cycle_id: str) -> int:
    try:
        return int(cycle_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _metric_label(metric_name: str) -> str:
    if metric_name == "pixel_aupimo_1e-5_1e-3":
        return "pixel_aupimo"
    return metric_name


def _labels(labels: dict[str, str]) -> str:
    return ",".join(f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items()))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.17g}"


def _openmetrics_timestamp(timestamp_ms: int) -> str:
    return f"{timestamp_ms / 1000:.3f}".rstrip("0").rstrip(".")


def _parse_timestamp(value: Any) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


if __name__ == "__main__":
    main()
