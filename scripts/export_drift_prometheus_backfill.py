"""Export historical drift observation artifacts as Prometheus OpenMetrics."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DRIFT_ROOT = Path(".cache/iqa/drift_observation")
DEFAULT_OUTPUT = Path(".cache/iqa/prometheus_backfill/drift_p4_observation.prom")
DRIFT_SCENARIO_DEFAULT = "production_replay_natural_piece_b_to_piece_a_p4_drift"
SOURCE_DOMAIN_DEFAULT = "piece_a_p4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-root", type=Path, default=DEFAULT_DRIFT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observation_dirs = sorted(
        {path.parent for path in args.drift_root.rglob("windows.jsonl")},
        key=lambda path: path.stat().st_mtime,
    )
    samples: list[dict[str, Any]] = []
    for observation_dir in observation_dirs:
        samples.extend(_drift_observation_samples(observation_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(_openmetrics(samples))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runs": len(observation_dirs),
                "samples": len(samples),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _drift_observation_samples(observation_dir: Path) -> list[dict[str, Any]]:
    summary = _read_json(observation_dir / "summary.json")
    windows = _read_jsonl(observation_dir / "windows.jsonl")
    scenario_id = str(summary.get("scenario_id") or DRIFT_SCENARIO_DEFAULT)
    observation_run_id = str(summary.get("run_id") or observation_dir.name)
    source_domain = SOURCE_DOMAIN_DEFAULT
    first_confirmed = _int_or_none(summary.get("first_confirmed_window_index"))
    active_models = _active_model_payloads(summary)
    output: list[dict[str, Any]] = []

    for window in windows:
        timestamp_ms = _parse_timestamp(window.get("evaluated_at"))
        if timestamp_ms is None:
            continue
        window_index = _int_or_none(window.get("window_index")) or 0
        base_labels = {
            "scenario_id": scenario_id,
            "source_domain": source_domain,
            "observation_run_id": observation_run_id,
        }
        status = str(window.get("status") or "clear")
        if status not in {"clear", "suspected", "confirmed"}:
            status = "clear"
        for status_name in ("clear", "suspected", "confirmed"):
            labels = dict(base_labels)
            labels["status"] = status_name
            output.append(_sample("iqa_drift_status", labels, 1 if status == status_name else 0, timestamp_ms))
        output.append(
            _sample(
                "iqa_drift_trigger_lifecycle",
                dict(base_labels),
                1 if bool(window.get("drift_confirmed")) else 0,
                timestamp_ms,
            )
        )
        metrics = window.get("metrics") or {}
        metric_specs = (
            ("drift_score", "iqa_drift_score"),
            ("window_events", "iqa_drift_window_events"),
            ("alert_rate", "iqa_drift_alert_rate"),
            ("red_rate", "iqa_drift_red_rate"),
            ("unexpected_red_rate", "iqa_drift_unexpected_red_rate"),
            ("roi_mask_nn_distance", "iqa_drift_roi_mask_nn_distance"),
            ("roi_mask_novelty_rate", "iqa_drift_roi_mask_novelty_rate"),
            ("roi_area_ratio", "iqa_drift_roi_area_ratio"),
            ("context_events_total", "iqa_drift_context_events_total"),
            ("roi_fail_rate", "iqa_drift_roi_fail_rate"),
            ("oracle_fn_rate", "iqa_drift_oracle_fn_rate"),
            ("domain_ratio", "iqa_drift_domain_ratio"),
            ("domain_score", "iqa_drift_domain_score"),
            ("degradation_score", "iqa_drift_degradation_score"),
        )
        output.append(_sample("iqa_drift_window_index", dict(base_labels), window_index, timestamp_ms))
        if first_confirmed is not None and window_index >= first_confirmed:
            output.append(_sample("iqa_drift_first_confirmed_window", dict(base_labels), first_confirmed, timestamp_ms))
        for source_name, metric_name in metric_specs:
            value = _finite_float(metrics.get(source_name))
            if value is not None:
                output.append(_sample(metric_name, dict(base_labels), value, timestamp_ms))
        for role, model_payload in active_models.items():
            labels = dict(base_labels)
            labels.update(
                {
                    "role": role,
                    "version": str(model_payload.get("version") or ""),
                    "registry_model_name": str(model_payload.get("registry_model_name") or ""),
                    "registered_model_version": str(model_payload.get("registered_model_version") or ""),
                    "registry_stage": str(model_payload.get("registry_stage") or ""),
                    "runtime_contract_status": str(model_payload.get("runtime_contract_status") or ""),
                }
            )
            output.append(_sample("iqa_drift_active_model_info", labels, 1, timestamp_ms))
    return output


def _active_model_payloads(summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for role, key in (
        ("classification", "active_classification_runtime"),
        ("localization", "active_localization_runtime"),
    ):
        payload = summary.get(key)
        if isinstance(payload, dict):
            output[role] = {str(name): str(value or "") for name, value in payload.items()}
    return output


def _openmetrics(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# TYPE iqa_drift_active_model_info gauge",
        "# TYPE iqa_drift_alert_rate gauge",
        "# TYPE iqa_drift_context_events_total gauge",
        "# TYPE iqa_drift_degradation_score gauge",
        "# TYPE iqa_drift_domain_ratio gauge",
        "# TYPE iqa_drift_domain_score gauge",
        "# TYPE iqa_drift_first_confirmed_window gauge",
        "# TYPE iqa_drift_oracle_fn_rate gauge",
        "# TYPE iqa_drift_red_rate gauge",
        "# TYPE iqa_drift_roi_area_ratio gauge",
        "# TYPE iqa_drift_roi_fail_rate gauge",
        "# TYPE iqa_drift_roi_mask_nn_distance gauge",
        "# TYPE iqa_drift_roi_mask_novelty_rate gauge",
        "# TYPE iqa_drift_score gauge",
        "# TYPE iqa_drift_status gauge",
        "# TYPE iqa_drift_trigger_lifecycle gauge",
        "# TYPE iqa_drift_unexpected_red_rate gauge",
        "# TYPE iqa_drift_window_events gauge",
        "# TYPE iqa_drift_window_index gauge",
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _labels(labels: dict[str, str]) -> str:
    return ",".join(f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items()))


def _escape_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


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


def _finite_float(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
