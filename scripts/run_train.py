"""Run the IQA training boundary (GPU-locked).

This is the ``train`` task of ``iqa_lifecycle`` run as a container on the ``ml``
image (ADR 0008, issue 09). It holds the single-GPU lock for its whole duration
(shared with the inference service and the eval task) so only one GPU consumer
runs at a time; pair it with the ``iqa_gpu`` Airflow pool (slots=1).

Scope note: this is a ``validated-summary`` boundary -- it acquires the lock and
reports the checkpoint reference it *would* produce, but does NOT yet run the real
training nor persist the checkpoint to MinIO (``persisted: false``). The heavy
training + checkpoint/metrics materialisation is tracked separately (issue 20),
mirroring the ingestion/dataset split (issues 18, 19).
"""

from __future__ import annotations

import argparse
import sys

from iqa.runtime import GpuBusyError, gpu_lock
from scripts.airflow_contracts import print_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--dataset-version", default="")
    parser.add_argument("--output-checkpoint", default="models/feature_ae/candidate.pt")
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


def _summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "service": "iqa-trainer",
        "stage": "train",
        "scenario_id": args.scenario_id,
        "dataset_version": args.dataset_version or None,
        "checkpoint": args.output_checkpoint,
        # Real training + MinIO checkpoint materialisation are deferred to issue 20.
        "persisted": False,
        "status": "validated",
    }


def main() -> None:
    args = parse_args()
    if args.no_gpu_lock:
        print_json(_summary(args))
        return
    try:
        with gpu_lock(owner="iqa-trainer", blocking=args.wait_for_gpu):
            print_json(_summary(args))
    except GpuBusyError as exc:
        print(f"iqa-trainer: {exc}", file=sys.stderr)
        raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
