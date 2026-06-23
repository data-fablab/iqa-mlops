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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(render_report(args.run_dir, show_epochs=args.epochs, show_cache=args.cache))


def render_report(run_dir: Path, *, show_epochs: bool = False, show_cache: bool = False) -> str:
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
        "pixel_ap",
        "unstable",
        "gate",
        "registry",
    )
    if show_cache:
        header = header + ("cache", "hit")
    rows = [header, *[_row(cycle, show_cache=show_cache) for cycle in cycles]]
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


def _row(cycle: dict[str, Any], *, show_cache: bool = False) -> tuple[str, ...]:
    active_value = cycle.get("active_metric_value")
    candidate_value = cycle.get("candidate_metric_value", cycle.get("selected_metric_value"))
    delta = cycle.get("metric_delta")
    candidate_metrics = cycle.get("candidate_metrics_on_eval_set") or cycle.get("metrics") or {}
    pixel_ap = candidate_metrics.get("pixel_ap")
    stability = cycle.get("candidate_aupimo_stability") or cycle.get("aupimo_stability") or {}
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
        "" if pixel_ap is None else f"{float(pixel_ap):.6g}",
        "yes" if stability.get("aupimo_unstable") else "no",
        str(cycle.get("gate_decision") or ""),
        registry,
    )
    if show_cache:
        row = row + (str(cycle.get("cache_status") or ""), str(cycle.get("cache_hit") or ""))
    return row


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
                f"pixel_ap={metrics.get('pixel_ap')} "
                f"predictions={item.get('predictions_path')}"
            )
    return lines if found else []


if __name__ == "__main__":
    main()
