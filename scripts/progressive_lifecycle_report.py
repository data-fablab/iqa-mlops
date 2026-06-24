"""Print a compact report for a progressive replay lifecycle run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", action="store_true", help="Print per-epoch metric history when available.")
    parser.add_argument("--cache", action="store_true", help="Print prediction cache status when available.")
    parser.add_argument("--mlflow", action="store_true", help="Print MLflow dataset/model logging evidence when available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(render_report(args.run_dir, show_epochs=args.epochs, show_cache=args.cache, show_mlflow=args.mlflow))


def render_report(run_dir: Path, *, show_epochs: bool = False, show_cache: bool = False, show_mlflow: bool = False) -> str:
    cycles_path = run_dir / "cycles.jsonl"
    if not cycles_path.is_file():
        progress_path = run_dir / "progress.json"
        if not progress_path.is_file():
            raise FileNotFoundError(f"progressive lifecycle report requires {cycles_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        return (
            "No completed cycles yet. "
            f"phase={progress.get('phase', 'unknown')} "
            f"active_model={progress.get('active_model_version', '')} "
            f"events={progress.get('events_processed', 0)} "
            f"lots={progress.get('lots_processed', 0)}"
        )
    cycles = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = (
        "cycle",
        "active_before",
        "candidate",
        "eval_n",
        "epoch",
        "active_aupimo",
        "candidate_aupimo",
        "delta",
        "loc_gate",
        "active_fn",
        "candidate_fn",
        "active_recall",
        "candidate_recall",
        "active_good_red",
        "candidate_good_red",
        "fn_delta",
        "good_red_delta",
        "class_gate",
        "class_progress",
        "gate",
        "reason",
        "registry",
    )
    if show_cache:
        header = header + ("active_cache", "candidate_cache", "aupimo_s", "pixel_s")
    if show_mlflow:
        header = header + ("run_id", "dataset", "model")
    rows = [header, *[_row(cycle, show_cache=show_cache, show_mlflow=show_mlflow) for cycle in cycles]]
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    report = "\n".join(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)).rstrip()
        for row in rows
    )
    if show_epochs:
        epoch_lines = _epoch_lines(cycles)
        if epoch_lines:
            report = report + "\n\n" + "\n".join(epoch_lines)
    return report


def _row(cycle: dict[str, Any], *, show_cache: bool = False, show_mlflow: bool = False) -> tuple[str, ...]:
    active_value = cycle.get("active_metric_value")
    candidate_value = cycle.get("candidate_metric_value", cycle.get("selected_metric_value"))
    delta = cycle.get("metric_delta")
    candidate_metrics = cycle.get("candidate_metrics_on_eval_set") or cycle.get("metrics") or {}
    active_metrics = cycle.get("active_metrics_on_eval_set") or {}
    active_false_negatives = cycle.get("active_false_negatives", active_metrics.get("false_negatives"))
    candidate_false_negatives = cycle.get("candidate_false_negatives", candidate_metrics.get("false_negatives"))
    active_good_red_count = cycle.get("active_good_red_count", active_metrics.get("good_red_count"))
    candidate_good_red_count = cycle.get("candidate_good_red_count", candidate_metrics.get("good_red_count"))
    fn_delta = cycle.get("fn_delta")
    good_red_delta = cycle.get("good_red_delta")
    localization_gate = cycle.get("localization_gate") or {}
    classification_gate = cycle.get("classification_gate") or {}
    classification_progress = cycle.get("classification_progress") or {}
    active_recall = classification_gate.get("active_image_recall", active_metrics.get("image_recall"))
    candidate_recall = classification_gate.get("candidate_image_recall", candidate_metrics.get("image_recall"))
    registry = cycle.get("registry_alias") or cycle.get("registry_stage") or ""
    if cycle.get("registered_model_version"):
        registry = f"{registry}:v{cycle['registered_model_version']}"
    elif cycle.get("registry_status") in {"failed", "not_registered", "skipped"}:
        registry = str(cycle.get("registry_status") or "")
    row = (
        str(cycle.get("cycle_id") or ""),
        str(cycle.get("active_model_before") or ""),
        str(cycle.get("candidate_version") or ""),
        str(cycle.get("evaluation_seen_events") or cycle.get("seen_events") or ""),
        str(cycle.get("selected_epoch") or ""),
        "" if active_value is None else f"{float(active_value):.6g}",
        "" if candidate_value is None else f"{float(candidate_value):.6g}",
        "" if delta is None else f"{float(delta):+.6g}",
        _gate_status(localization_gate.get("passed")),
        "" if active_false_negatives is None else f"{float(active_false_negatives):.0f}",
        "" if candidate_false_negatives is None else f"{float(candidate_false_negatives):.0f}",
        "" if active_recall is None else f"{float(active_recall):.3f}",
        "" if candidate_recall is None else f"{float(candidate_recall):.3f}",
        "" if active_good_red_count is None else f"{float(active_good_red_count):.0f}",
        "" if candidate_good_red_count is None else f"{float(candidate_good_red_count):.0f}",
        "" if fn_delta is None else f"{float(fn_delta):+.0f}",
        "" if good_red_delta is None else f"{float(good_red_delta):+.0f}",
        _gate_status(classification_gate.get("passed")),
        str(classification_progress.get("summary") or ""),
        str(cycle.get("gate_decision") or ""),
        str(cycle.get("gate_reason") or ""),
        registry,
    )
    if show_cache:
        timings = cycle.get("candidate_metric_timings") or {}
        row = row + (
            str(cycle.get("active_cache_status") or cycle.get("cache_status") or ""),
            str(cycle.get("candidate_cache_status") or ""),
            _duration(timings.get("aupimo_compute_seconds")),
            _duration(timings.get("pixel_rank_metrics_seconds")),
        )
    if show_mlflow:
        row = row + (
            str(cycle.get("mlflow_run_id") or ""),
            _bool_status(cycle.get("mlflow_dataset_logged")),
            _bool_status(cycle.get("mlflow_model_logged")),
        )
    return row


def _bool_status(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def _gate_status(value: Any) -> str:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return ""


def _duration(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _epoch_lines(cycles: list[dict[str, Any]]) -> list[str]:
    lines = ["epoch metrics"]
    found = False
    for cycle in cycles:
        history = cycle.get("epoch_metric_history") or []
        for item in history:
            metrics = item.get("metrics") or {}
            found = True
            lines.append(
                f"{cycle.get('cycle_id')} epoch={item.get('epoch')} "
                f"aupimo={metrics.get('pixel_aupimo_1e-5_1e-3')} "
                f"pixel_ap={metrics.get('pixel_ap')}"
            )
    return lines if found else []


if __name__ == "__main__":
    main()
