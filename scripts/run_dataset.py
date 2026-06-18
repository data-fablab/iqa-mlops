"""Run the IQA candidate-dataset boundary.

This is the ``dataset`` task of ``iqa_lifecycle`` run as a container (ADR 0008,
issue 08): it validates the candidate manifest **inside the data image** and
materialises it into the object store (MinIO/S3) so the downstream train task can
resolve the candidate dataset by URI rather than a repo path (issue 19).

The object-store backend is the env-driven one (``IQA_OBJECT_STORE_BACKEND``,
``memory`` by default, ``s3`` for MinIO): the boundary stays torch-free and the
write target is a deploy decision, not a code change (ADR 0008).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iqa.storage import IQA_BUCKETS
from iqa.storage.object_store import ObjectStore, create_object_store

from scripts.airflow_contracts import csv_manifest_summary, print_json_with_xcom_ref

DATASET_BUCKET = IQA_BUCKETS["source_datasets"]


def dataset_key(scenario_id: str, candidate_version: str, manifest: Path) -> str:
    """Deterministic object key for a candidate dataset manifest."""
    version = candidate_version or "candidate"
    return f"model_datasets/{scenario_id}/{version}/{manifest.name}"


def materialise_dataset(
    store: ObjectStore,
    *,
    manifest: Path,
    scenario_id: str,
    candidate_version: str,
) -> str:
    """Write ``manifest`` to the object store and return its ``s3://`` URI."""
    key = dataset_key(scenario_id, candidate_version, manifest)
    return store.put_bytes(
        DATASET_BUCKET, key, manifest.read_bytes(), content_type="text/csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/model_datasets/feature_ae_good_v002.csv"),
    )
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--candidate-version", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = csv_manifest_summary(args.manifest, label="dataset manifest")
    dataset_uri = materialise_dataset(
        create_object_store(),
        manifest=args.manifest,
        scenario_id=args.scenario_id,
        candidate_version=args.candidate_version,
    )
    result = {
        "service": "iqa-dataset",
        "scenario_id": args.scenario_id,
        "candidate_version": args.candidate_version or None,
        "manifest": manifest,
        "materialized": True,
        "dataset_uri": dataset_uri,
        "status": "materialized",
    }
    # The dataset URI is the XCom the downstream train task pulls (ADR 0008).
    print_json_with_xcom_ref(result, dataset_uri)


if __name__ == "__main__":
    main()
