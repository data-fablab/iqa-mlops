"""Run the IQA ingestion batch boundary.

The ``ingestion`` task validates the piece-event manifest **inside the data
image** and materialises it into the object store (MinIO/S3) so the ingested
batch is addressable by URI rather than a repo path (ADR 0008, issue 18).

The object-store backend is the env-driven one (``IQA_OBJECT_STORE_BACKEND``,
``memory`` by default, ``s3`` for MinIO): the boundary stays torch-free and the
write target is a deploy decision, not a code change (ADR 0008).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iqa.storage import IQA_BUCKETS
from iqa.storage.object_store import ObjectStore, create_object_store

from scripts.airflow_contracts import csv_manifest_summary, print_json

INGESTED_BUCKET = IQA_BUCKETS["ingested_images"]


def ingestion_key(scenario_id: str, source: str, manifest: Path) -> str:
    """Deterministic object key for an ingested piece-event manifest."""
    return f"ingested/{scenario_id}/{source}/{manifest.name}"


def materialise_ingestion(
    store: ObjectStore,
    *,
    manifest: Path,
    scenario_id: str,
    source: str,
) -> str:
    """Write ``manifest`` to the object store and return its ``s3://`` URI."""
    key = ingestion_key(scenario_id, source, manifest)
    return store.put_bytes(
        INGESTED_BUCKET, key, manifest.read_bytes(), content_type="text/csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/metadata/casting_piece_events.csv"))
    parser.add_argument("--source", choices=["historical_replay", "production_ingest"], default="historical_replay")
    parser.add_argument("--scenario-id", default="raw_ingestion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = csv_manifest_summary(args.manifest, label="ingestion manifest")
    ingested_uri = materialise_ingestion(
        create_object_store(),
        manifest=args.manifest,
        scenario_id=args.scenario_id,
        source=args.source,
    )
    result = {
        "service": "iqa-ingestion",
        "source": args.source,
        "scenario_id": args.scenario_id,
        "manifest": manifest,
        "materialized": True,
        "ingested_uri": ingested_uri,
        "status": "ingested",
    }
    print_json(result)


if __name__ == "__main__":
    main()
