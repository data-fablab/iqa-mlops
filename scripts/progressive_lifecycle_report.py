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
        raise FileNotFoundError(f"progressive lifecycle report requires {cycles_path}")
    cycles = [json.loads(line) for line in cycles_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [
        ("cycle", "model", "selected_metric", "value", "gate", "stage", "mlflow"),
        *[_row(cycle) for cycle in cycles],
    ]
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    return "\n".join(
        "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)).rstrip()
        for row in rows
    )


def _row(cycle: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    value = cycle.get("selected_metric_value")
    return (
        str(cycle.get("cycle_id") or ""),
        str(cycle.get("candidate_version") or ""),
        str(cycle.get("selected_metric") or ""),
        "" if value is None else f"{float(value):.6g}",
        str(cycle.get("gate_decision") or ""),
        str(cycle.get("registry_stage") or ""),
        str(cycle.get("mlflow_run_id") or ""),
    )


if __name__ == "__main__":
    main()
