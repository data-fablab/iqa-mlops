"""S3 URI contracts for the local IQA MinIO storage."""

from __future__ import annotations

from dataclasses import dataclass


IQA_BUCKETS = {
    "source_datasets": "iqa-source-datasets",
    "dvc": "iqa-dvc",
    "ingested_images": "iqa-ingested-images",
    "mlflow": "mlflow-artifacts",
    "roi_masks": "iqa-roi-masks",
    "heatmaps": "iqa-heatmaps",
    "gt_masks": "iqa-gt-masks",
    "models": "iqa-models",
    "backups": "iqa-backups",
}


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def parse_s3_uri(uri: str) -> S3Uri:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}.")

    bucket_and_key = uri.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Expected an S3 URI with bucket and key, got {uri!r}.")
    return S3Uri(bucket=bucket, key=key)


__all__ = ["IQA_BUCKETS", "S3Uri", "parse_s3_uri"]
