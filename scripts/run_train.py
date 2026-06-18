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
import csv
import io

from iqa.storage import parse_s3_uri
from iqa.storage.object_store import ObjectStore

from scripts.gpu_boundary import emit_gpu_locked_summary


def resolve_dataset(store: ObjectStore, dataset_uri: str) -> dict[str, object]:
    """Fetch the candidate dataset manifest the dataset task wrote, by URI.

    Closes the producer->consumer loop (issues 19/20): the dataset boundary
    materialised the manifest into the object store; train resolves it here
    instead of reading a repo path. Returns the URI and the row count it will
    train on. Raises ``KeyError`` if the URI is absent from the store.
    """
    location = parse_s3_uri(dataset_uri)
    body = store.get_bytes(location.bucket, location.key)
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8"))))
    return {"uri": dataset_uri, "rows": len(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--dataset-version", default="")
    parser.add_argument(
        "--dataset-uri",
        default="",
        help="s3:// URI of the candidate dataset materialised by the dataset task.",
    )
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
        "dataset_uri": args.dataset_uri or None,
        "checkpoint": args.output_checkpoint,
        # Real training + MinIO checkpoint materialisation are deferred to issue 20.
        "persisted": False,
        "status": "validated",
    }


def main() -> None:
    args = parse_args()
    emit_gpu_locked_summary(
        owner="iqa-trainer",
        summary=_summary(args),
        no_gpu_lock=args.no_gpu_lock,
        wait_for_gpu=args.wait_for_gpu,
    )


if __name__ == "__main__":
    main()
