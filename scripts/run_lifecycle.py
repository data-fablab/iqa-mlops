"""Run the IQA lifecycle batch boundary."""

from __future__ import annotations

import argparse
import json
import sys

from iqa.registry import ModelRegistryRef, registered_model_name
from iqa.runtime import GpuBusyError, gpu_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--stage", default="candidate")
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="Block until the GPU lock is free instead of refusing immediately.",
    )
    parser.add_argument(
        "--no-gpu-lock",
        action="store_true",
        help="Skip the GPU lock (CPU-only dry run; never use during a live demo).",
    )
    return parser.parse_args()


def _run_lifecycle(args: argparse.Namespace) -> dict[str, object]:
    return {
        "service": "iqa-trainer",
        "lifecycle": "train_eval_gate_promote",
        "registry": ModelRegistryRef(
            scenario_id=args.scenario_id,
            registered_model_name=registered_model_name(args.scenario_id),
            stage=args.stage,
        ).to_dict(),
        "status": "planned",
    }


def main() -> None:
    args = parse_args()
    if args.no_gpu_lock:
        print(json.dumps(_run_lifecycle(args), indent=2, sort_keys=True))
        return
    try:
        with gpu_lock(owner="iqa-trainer", blocking=args.wait_for_gpu):
            print(json.dumps(_run_lifecycle(args), indent=2, sort_keys=True))
    except GpuBusyError as exc:
        print(f"iqa-trainer: {exc}", file=sys.stderr)
        raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
