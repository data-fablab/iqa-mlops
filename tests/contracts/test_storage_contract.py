from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqa.storage import IQA_BUCKETS, parse_s3_uri

ROOT = Path(".")


def test_iqa_minio_bucket_names() -> None:
    assert IQA_BUCKETS == {
        "source_datasets": "iqa-source-datasets",
        "dvc": "iqa-dvc",
        "ingested_images": "iqa-ingested-images",
        "mlflow": "mlflow-artifacts",
        "roi_masks": "iqa-roi-masks",
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


def test_model_manifests_reference_minio_artifacts() -> None:
    for manifest_path in sorted((ROOT / "models" / "manifests").glob("*/model_manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact_uri = payload["artifact_uri"]
        parsed = parse_s3_uri(artifact_uri)

        assert parsed.bucket == "iqa-models"
        assert parsed.key.endswith("/checkpoint.pt")


def test_readme_documents_model_artifact_restore_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "iqa-restore-model-artifacts" in readme
    assert ".cache/iqa/models" in readme
