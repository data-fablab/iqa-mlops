"""Export P4 correction lifecycle metrics aligned to the drift story timeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.export_lifecycle_prometheus_backfill import _openmetrics, _promotion_selection_samples


SCENARIO_ID = "production_replay_natural_piece_b_to_piece_a_p4_drift"
DEFAULT_DRIFT_ROOT = Path(".cache/iqa/drift_observation") / SCENARIO_ID
DEFAULT_LIFECYCLE_ROOT = Path(".cache/iqa/replay_lifecycle") / SCENARIO_ID
DEFAULT_OUTPUT = Path(".cache/iqa/prometheus_backfill/drift_p4_correction_story.prom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drift-root", type=Path, default=DEFAULT_DRIFT_ROOT)
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--after-confirmed-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drift_dir, confirmed_at_ms = _latest_confirmed_drift(args.drift_root)
    lifecycle_dir = _latest_completed_lifecycle(args.lifecycle_root)
    samples = _promotion_selection_samples(lifecycle_dir)
    if not samples:
        raise RuntimeError(f"no lifecycle samples exported from {lifecycle_dir}")

    source_anchor_ms = max(int(sample["timestamp_ms"]) for sample in samples)
    target_anchor_ms = confirmed_at_ms + args.after_confirmed_seconds * 1000
    shift_ms = target_anchor_ms - source_anchor_ms
    shifted = []
    for sample in samples:
        shifted_sample = dict(sample)
        shifted_sample["timestamp_ms"] = int(sample["timestamp_ms"]) + shift_ms
        shifted.append(shifted_sample)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(_openmetrics(shifted))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "drift_observation_dir": str(drift_dir),
                "lifecycle_run_dir": str(lifecycle_dir),
                "confirmed_at_ms": confirmed_at_ms,
                "target_anchor_ms": target_anchor_ms,
                "shift_ms": shift_ms,
                "samples": len(shifted),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _latest_confirmed_drift(root: Path) -> tuple[Path, int]:
    candidates: list[tuple[float, Path, int]] = []
    for summary_path in root.rglob("summary.json"):
        summary = _read_json(summary_path)
        if not bool(summary.get("drift_confirmed") or summary.get("ever_confirmed")):
            continue
        confirmed_at_ms = _first_confirmed_timestamp(summary_path.parent / "windows.jsonl")
        if confirmed_at_ms is not None:
            candidates.append((summary_path.parent.stat().st_mtime, summary_path.parent, confirmed_at_ms))
    if not candidates:
        raise RuntimeError(f"no confirmed drift observation found under {root}")
    _, path, timestamp_ms = max(candidates, key=lambda item: item[0])
    return path, timestamp_ms


def _latest_completed_lifecycle(root: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for progress_path in root.rglob("progress.json"):
        run_dir = progress_path.parent
        progress = _read_json(progress_path)
        cycles = _read_jsonl(run_dir / "cycles.jsonl")
        if int(progress.get("cycles_completed") or len(cycles)) <= 0:
            continue
        if not cycles:
            continue
        candidates.append((run_dir.stat().st_mtime, run_dir))
    if not candidates:
        raise RuntimeError(f"no completed lifecycle run found under {root}")
    return max(candidates, key=lambda item: item[0])[1]


def _first_confirmed_timestamp(windows_path: Path) -> int | None:
    for window in _read_jsonl(windows_path):
        if bool(window.get("drift_confirmed")) or str(window.get("status") or "") == "confirmed":
            timestamp_ms = _parse_timestamp(window.get("evaluated_at"))
            if timestamp_ms is not None:
                return timestamp_ms
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_timestamp(value: Any) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


if __name__ == "__main__":
    main()
