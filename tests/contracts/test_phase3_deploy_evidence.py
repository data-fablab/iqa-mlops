from pathlib import Path

import yaml

from scripts.check_deploy_evidence import build_deploy_evidence

ROOT = Path(".")


def test_phase3_deploy_evidence_command_is_public() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'iqa-check-deploy-evidence = "scripts.check_deploy_evidence:main"' in pyproject


def test_phase3_deploy_evidence_static_checks_pass() -> None:
    evidence = build_deploy_evidence()

    assert evidence["status"] == "validated"
    assert evidence["image_tag_source"] == "IQA_IMAGE_TAG"
    assert evidence["registry_source"] == "IQA_IMAGE_REGISTRY"
    assert evidence["recommended_tag_strategy"] == "ci_sha"
    assert set(evidence["published_services"]) >= {
        "iqa-api",
        "iqa-inference",
        "iqa-dvc-gate",
        "iqa-trainer",
        "airflow-webserver",
        "airflow-scheduler",
    }


def test_prod_compose_uses_tagged_registry_images_without_latest_or_build_fallback() -> None:
    prod = yaml.safe_load((ROOT / "deploy" / "docker-compose.prod.yml").read_text(encoding="utf-8"))
    services = prod["services"]

    for service in [
        "iqa-api",
        "iqa-inference",
        "iqa-ingestion",
        "iqa-replay",
        "iqa-trainer",
        "iqa-monitoring",
        "iqa-dvc-gate",
        "airflow-init",
        "airflow-webserver",
        "airflow-scheduler",
    ]:
        image = services[service]["image"]
        assert "IQA_IMAGE_REGISTRY" in image
        assert "IQA_IMAGE_TAG" in image
        assert ":latest" not in image
    services_with_base_build = [
        "iqa-api",
        "iqa-inference",
        "iqa-ingestion",
        "iqa-replay",
        "iqa-trainer",
        "iqa-monitoring",
        "iqa-dvc-gate",
        "airflow-init",
    ]
    for service in services_with_base_build:
        assert services[service]["build"] is None
    for service in ["airflow-webserver", "airflow-scheduler"]:
        assert "build" not in services[service]
        assert services[service]["environment"]["IQA_IMAGE_DATA"].endswith("/iqa-data:${IQA_IMAGE_TAG:-v0.1.0}")
        assert services[service]["environment"]["IQA_IMAGE_ML"].endswith("/iqa-ml:${IQA_IMAGE_TAG:-v0.1.0}")


def test_ci_publish_images_covers_role_and_airflow_images_without_latest() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    for expected in ["iqa-serving", "iqa-ml", "iqa-data", "iqa-dvc-gate", "iqa-airflow"]:
        assert expected in workflow
    assert "flavor: latest=false" in workflow
    assert "IQA_PUBLISH_IMAGES" in workflow
    assert "IQA_IMAGE_REGISTRY" in workflow
    assert "DOCKERHUB_USERNAME" in workflow
    assert "DOCKERHUB_TOKEN" in workflow


def test_deploy_docs_cover_pull_up_smoke_gateway_and_registry_boundaries() -> None:
    docs = "\n".join(
        [
            (ROOT / "docs" / "phase3-deploy-evidence.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "deploy_runbook.md").read_text(encoding="utf-8"),
        ]
    )

    for expected in [
        "docker compose -f docker-compose.yml -f docker-compose.prod.yml pull",
        "docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d",
        "Docker Hub",
        "IQA_PUBLISH_IMAGES",
        "IQA_IMAGE_REGISTRY",
        "IQA_IMAGE_TAG",
        "IQA_IMAGE_TAG=sha-<commit>",
        "DOCKERHUB_USERNAME",
        "DOCKERHUB_TOKEN",
        "gh variable set",
        "gh secret set",
        "docker login",
        "iqa-dvc-gate",
        "Kong",
        "Nginx",
        "MLflow Registry",
        "pull -> up -d -> smoke",
    ]:
        assert expected in docs


def test_smoke_test_covers_gateway_core_services_and_observability() -> None:
    smoke = (ROOT / "deploy" / "smoke-test.sh").read_text(encoding="utf-8")

    for expected in [
        "api /health",
        "inference /health",
        "minio live",
        "model/version?scenario_id=production_replay_natural",
        "mlflow up",
        "grafana health",
        "airflow health",
        "gateway -> api",
        "gateway -> grafana",
        "gateway -> airflow",
        "gateway -> mlflow",
    ]:
        assert expected in smoke
