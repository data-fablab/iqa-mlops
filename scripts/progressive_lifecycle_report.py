"""Print a compact report for a progressive replay lifecycle run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(render_report(args.run_dir))


def render_report(run_dir: Path) -> str:
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
    rows = [
        (
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
        ),
        *[_row(cycle) for cycle in cycles],
    ]
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)).rstrip()
        for row in rows
    )


def _row(cycle: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str]:
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
    return (
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


if __name__ == "__main__":
    main()
