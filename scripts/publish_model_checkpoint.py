"""Publish a model checkpoint to MinIO/S3 and update its local manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from iqa.storage.artifacts import sha256_file
from iqa.storage.uris import parse_s3_uri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--source-run", default="")
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--usage-note", default="")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--create-bucket", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_env_file(args.env_file)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {args.manifest}")

    checksum = sha256_file(args.checkpoint)
    parsed = parse_s3_uri(args.artifact_uri)
    client = _build_s3_client()
    if args.create_bucket:
        _ensure_bucket(client, parsed.bucket)
    client.upload_file(str(args.checkpoint), parsed.bucket, parsed.key)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload.update(
        {
            "model_version": args.model_version,
            "artifact_uri": args.artifact_uri,
            "sha256": checksum,
        }
    )
    if args.source_run:
        payload["source_run"] = args.source_run
    if args.source_checkpoint:
        payload["source_checkpoint"] = args.source_checkpoint
    if args.usage_note:
        payload["usage_note"] = args.usage_note
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_uri": args.artifact_uri,
                "bucket": parsed.bucket,
                "checkpoint": str(args.checkpoint),
                "key": parsed.key,
                "manifest": str(args.manifest),
                "model_version": args.model_version,
                "sha256": checksum,
                "status": "published",
            },
            indent=2,
            sort_keys=True,
        )
    )


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
        raise ImportError("boto3 is required to publish model checkpoints to S3/MinIO.") from exc

    endpoint_url = os.getenv("IQA_S3_ENDPOINT_URL")
    access_key = os.getenv("IQA_S3_ACCESS_KEY_ID") or os.getenv("MINIO_ROOT_USER") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("IQA_S3_SECRET_ACCESS_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key == "change-me" and os.getenv("MINIO_ROOT_USER"):
        access_key = os.getenv("MINIO_ROOT_USER")
    if secret_key == "change-me" and os.getenv("MINIO_ROOT_PASSWORD"):
        secret_key = os.getenv("MINIO_ROOT_PASSWORD")

    kwargs: dict[str, str] = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    if region := os.getenv("IQA_S3_REGION"):
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _ensure_bucket(client: Any, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        client.create_bucket(Bucket=bucket)


if __name__ == "__main__":
    main()
