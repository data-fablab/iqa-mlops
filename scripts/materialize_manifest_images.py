"""Materialize manifest images from MinIO/S3 into a local runtime cache."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from iqa.storage import IQA_BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/iqa/source_datasets/hss-iad"))
    parser.add_argument("--bucket", default=IQA_BUCKETS["source_datasets"])
    parser.add_argument("--key-prefix", default="hss-iad")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_file(args.env_file)
    client = _build_s3_client()
    relative_paths = _manifest_relative_paths(args.manifest)

    materialized = []
    for relative_path in relative_paths:
        destination = args.cache_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        key = _object_key(args.key_prefix, relative_path)
        if not (args.skip_existing and destination.is_file()):
            client.download_file(args.bucket, key, str(destination))
        materialized.append(str(destination))

    print(
        json.dumps(
            {
                "bucket": args.bucket,
                "cache_root": str(args.cache_root),
                "image_count": len(materialized),
                "image_root": str(args.cache_root),
                "key_prefix": args.key_prefix,
                "status": "materialized",
            },
            indent=2,
            sort_keys=True,
        )
    )


def _manifest_relative_paths(manifests: list[Path]) -> list[str]:
    paths: set[str] = set()
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                for column in ("relative_paths", "relative_path", "gt_mask_paths", "gt_mask_path", "mask_paths", "mask_path"):
                    values = row.get(column) or ""
                    paths.update(_normalize_manifest_path(item) for item in values.split("|") if item.strip())
    return sorted(paths)


def _normalize_manifest_path(value: str) -> str:
    path = value.strip().replace("\\", "/")
    for prefix in ("../raw/hss-iad/", "data/raw/hss-iad/", "raw/hss-iad/", "hss-iad/"):
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _object_key(prefix: str, relative_path: str) -> str:
    return f"{prefix.strip('/').replace('\\', '/')}/{relative_path.strip('/').replace('\\', '/')}"


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _build_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - dependency is present in normal envs.
        raise ImportError("boto3 is required to materialize images from S3/MinIO.") from exc

    access_key = os.getenv("IQA_S3_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("IQA_S3_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key == "change-me" and os.getenv("MINIO_ROOT_USER"):
        access_key = os.getenv("MINIO_ROOT_USER")
    if secret_key == "change-me" and os.getenv("MINIO_ROOT_PASSWORD"):
        secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("IQA_S3_ENDPOINT_URL"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv("IQA_S3_REGION", "us-east-1"),
    )


if __name__ == "__main__":
    main()
