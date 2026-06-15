"""Contract tests for MLflow Registry as source of truth (IQA1_KEN07)."""
from __future__ import annotations

import yaml
from pathlib import Path


def test_paths_yaml_defines_all_required_buckets() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))

    required_buckets = {
        "source_datasets_bucket",
        "ingested_images_bucket",
        "dvc_bucket",
        "heatmaps_bucket",
        "models_bucket",
        "mlflow_bucket",
    }
    storage = paths.get("storage", {})
    assert required_buckets <= set(storage.keys()), f"Missing buckets: {required_buckets - set(storage.keys())}"


def test_env_example_has_mlflow_source_of_truth_flag() -> None:
    env_content = Path(".env.example").read_text(encoding="utf-8")
    assert "IQA_MLFLOW_REGISTRY_SOURCE_OF_TRUTH=true" in env_content, (
        "IQA_MLFLOW_REGISTRY_SOURCE_OF_TRUTH flag missing or not set to true"
    )


def test_mlflow_registry_doc_exists() -> None:
    doc = Path("docs/MLflow-Registry.md")
    assert doc.exists(), "docs/MLflow-Registry.md not found"
    content = doc.read_text(encoding="utf-8")
    assert "source of truth" in content.lower() or "source de vérité" in content.lower()
    assert "ADR 0006" in content
    assert "ADR 0003" in content


def test_adr_0006_references_mlflow_registry_doc() -> None:
    adr006 = Path("docs/adr/0006-mlflow-registry-source-verite.md")
    content = adr006.read_text(encoding="utf-8")
    assert "MLflow-Registry.md" in content or "docs/MLflow-Registry.md" in content
