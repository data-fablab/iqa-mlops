from __future__ import annotations

import pytest

from iqa.storage import IQA_BUCKETS, parse_s3_uri


def test_iqa_minio_bucket_names() -> None:
    assert IQA_BUCKETS == {
        "source_datasets": "iqa-source-datasets",
        "dvc": "iqa-dvc",
        "ingested_images": "iqa-ingested-images",
        "mlflow": "mlflow-artifacts",
        "heatmaps": "iqa-heatmaps",
        "models": "iqa-models",
        "backups": "iqa-backups",
    }


def test_parse_s3_uri() -> None:
    parsed = parse_s3_uri("s3://iqa-models/prod/model_manifest.json")

    assert parsed.bucket == "iqa-models"
    assert parsed.key == "prod/model_manifest.json"


def test_parse_s3_uri_rejects_invalid_uri() -> None:
    with pytest.raises(ValueError, match="Expected an s3:// URI"):
        parse_s3_uri("models/prod/model.pt")

    with pytest.raises(ValueError, match="bucket and key"):
        parse_s3_uri("s3://iqa-models")
