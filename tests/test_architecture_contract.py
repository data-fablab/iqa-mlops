from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(".")


def test_target_iqa_packages_are_importable() -> None:
    for module_name in ["iqa.replay", "iqa.feedback", "iqa.monitoring", "iqa.registry"]:
        assert importlib.util.find_spec(module_name) is not None


def test_airflow_dags_are_present_and_importable() -> None:
    dag_paths = [
        ROOT / "airflow" / "dags" / "iqa_ingestion.py",
        ROOT / "airflow" / "dags" / "iqa_replay.py",
        ROOT / "airflow" / "dags" / "iqa_lifecycle.py",
        ROOT / "airflow" / "dags" / "iqa_monitoring.py",
    ]
    for path in dag_paths:
        assert path.is_file()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)


def test_deploy_compose_contains_target_services() -> None:
    compose = (ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8")

    for service in {
        "iqa-api",
        "iqa-inference",
        "iqa-streamlit",
        "iqa-ingestion",
        "iqa-replay",
        "iqa-trainer",
        "iqa-monitoring",
        "airflow-webserver",
        "airflow-scheduler",
        "mlflow",
        "minio",
        "minio-init",
        "postgres",
        "prometheus",
        "grafana",
        "reverse-proxy",
    }:
        assert f"  {service}:" in compose


def test_deploy_support_directories_exist() -> None:
    for path in [
        ROOT / "deploy" / "nginx" / "default.conf",
        ROOT / "deploy" / "minio" / "init-buckets.sh",
        ROOT / "deploy" / "prometheus" / "prometheus.yml",
        ROOT / "deploy" / "grafana" / "provisioning" / "README.md",
    ]:
        assert path.is_file()


def test_env_exposes_service_boundaries_and_databases() -> None:
    env = (ROOT / ".env.example").read_text(encoding="utf-8")

    for term in [
        "IQA_INFERENCE_URL=",
        "iqa_metadata",
        "IQA_MLFLOW_DB_URL=postgresql://",
        "IQA_AIRFLOW_DB_URL=postgresql://",
        "IQA_MLFLOW_REGISTRY_SOURCE_OF_TRUTH=true",
        "IQA_ADMIN_TOKEN=",
        "IQA_SERVICE_TOKEN=",
    ]:
        assert term in env
