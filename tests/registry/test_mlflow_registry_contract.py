"""Contract tests for MLflow Registry as source of truth (IQA1_KEN07)."""
from __future__ import annotations

import yaml
from pathlib import Path
from types import SimpleNamespace

from iqa.registry import mlflow_registry


def test_paths_yaml_defines_all_required_buckets() -> None:
    paths = yaml.safe_load(Path("configs/paths.yaml").read_text(encoding="utf-8"))

    required_buckets = {
        "source_datasets_bucket",
        "ingested_images_bucket",
        "dvc_bucket",
        "gt_masks_bucket",
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
    doc = Path("docs/mlflow-registry.md")
    assert doc.exists(), "docs/mlflow-registry.md not found"
    content = doc.read_text(encoding="utf-8")
    assert "source of truth" in content.lower() or "source de vérité" in content.lower()
    assert "ADR 0006" in content
    assert "ADR 0003" in content


def test_adr_0006_references_mlflow_registry_doc() -> None:
    adr006 = Path("docs/adr/0006-mlflow-registry-source-verite.md")
    content = adr006.read_text(encoding="utf-8")
    assert "mlflow-registry.md" in content or "docs/mlflow-registry.md" in content


def test_register_run_uses_iqa_tracking_uri_fallback(monkeypatch) -> None:
    calls: dict[str, list[object]] = {"tracking_uris": [], "client_uris": [], "downloads": [], "sources": []}

    class FakeClient:
        def __init__(self, tracking_uri: str | None = None) -> None:
            calls["client_uris"].append(tracking_uri)

        def get_run(self, run_id: str) -> SimpleNamespace:
            return SimpleNamespace(info=SimpleNamespace(artifact_uri="s3://mlflow-artifacts/0/run-001/artifacts"))

        def download_artifacts(self, run_id: str, path: str) -> str:
            calls["downloads"].append((run_id, path))
            return "/tmp/MLmodel"

        def create_registered_model(self, name: str) -> None:
            return None

        def create_model_version(self, name: str, source: str, run_id: str) -> SimpleNamespace:
            calls["sources"].append(source)
            return SimpleNamespace(version="7")

        def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
            return None

    class FakeMlflow:
        class exceptions:
            class MlflowException(Exception):
                pass

        class tracking:
            MlflowClient = FakeClient

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uris"].append(uri)

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("IQA_MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(mlflow_registry, "mlflow", FakeMlflow)

    result = mlflow_registry.register_run_to_model(
        run_id="run-001",
        scenario_id="production_replay_natural_train_v004",
        stage="test",
        model_artifact_path="model_classification",
    )

    assert calls["tracking_uris"] == ["http://mlflow:5000"]
    assert calls["client_uris"] == ["http://mlflow:5000"]
    assert calls["downloads"] == [("run-001", "model_classification/MLmodel")]
    assert calls["sources"] == ["s3://mlflow-artifacts/0/run-001/artifacts/model_classification"]
    assert result["version"] == "7"
    assert result["model_artifact_path"] == "model_classification"
