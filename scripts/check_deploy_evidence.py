"""Validate Phase 3 deploy-from-images evidence without starting Docker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_COMPOSE = ROOT / "deploy" / "docker-compose.yml"
PROD_COMPOSE = ROOT / "deploy" / "docker-compose.prod.yml"
SMOKE_TEST = ROOT / "deploy" / "smoke-test.sh"
DEPLOY_RUNBOOK = ROOT / "docs" / "deploy_runbook.md"
PHASE3_DEPLOY_DOC = ROOT / "docs" / "phase3-deploy-evidence.md"

PUBLISHED_IMAGE_SERVICES = {
    "iqa-api": "iqa-serving",
    "iqa-inference": "iqa-ml",
    "iqa-ingestion": "iqa-data",
    "iqa-replay": "iqa-data",
    "iqa-trainer": "iqa-ml",
    "iqa-monitoring": "iqa-data",
    "iqa-dvc-gate": "iqa-dvc-gate",
    "airflow-init": "iqa-airflow",
    "airflow-webserver": "iqa-airflow",
    "airflow-scheduler": "iqa-airflow",
}
EXPECTED_SMOKE_TERMS = [
    "gateway -> api",
    "gateway -> grafana",
    "gateway -> airflow",
    "gateway -> mlflow",
    "api /health",
    "inference /health",
    "minio live",
    "grafana health",
    "airflow health",
    "mlflow up",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable evidence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = build_deploy_evidence()
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Phase 3 deploy evidence checks passed.")


def build_deploy_evidence() -> dict[str, Any]:
    base = _read_yaml(BASE_COMPOSE)
    prod = _read_yaml(PROD_COMPOSE)
    base_services = _services(base)
    prod_services = _services(prod)
    smoke = SMOKE_TEST.read_text(encoding="utf-8")
    runbook = DEPLOY_RUNBOOK.read_text(encoding="utf-8")
    phase3_doc = PHASE3_DEPLOY_DOC.read_text(encoding="utf-8")

    for service, image_name in PUBLISHED_IMAGE_SERVICES.items():
        if service not in base_services:
            raise AssertionError(f"missing base service: {service}")
        service_config = prod_services.get(service)
        if not isinstance(service_config, dict):
            raise AssertionError(f"missing prod override for published service: {service}")
        image = str(service_config.get("image") or "")
        if not image:
            raise AssertionError(f"prod service has no image: {service}")
        if ":latest" in image or image.endswith(":latest"):
            raise AssertionError(f"prod service uses latest tag: {service} -> {image}")
        if image_name not in image:
            raise AssertionError(f"prod service image does not reference {image_name}: {service} -> {image}")
        if "IQA_IMAGE_TAG" not in image:
            raise AssertionError(f"prod service image is not controlled by IQA_IMAGE_TAG: {service}")
        base_has_build = isinstance(base_services.get(service), dict) and "build" in base_services[service]
        if base_has_build and service_config.get("build", "not-overridden") is not None:
            raise AssertionError(f"prod service keeps a build fallback: {service}")
        if not base_has_build and "build" in service_config:
            raise AssertionError(f"prod service adds an invalid build override: {service}")

    for airflow_service in ["airflow-webserver", "airflow-scheduler"]:
        env = prod_services[airflow_service].get("environment", {})
        if "iqa-data" not in str(env.get("IQA_IMAGE_DATA", "")):
            raise AssertionError(f"{airflow_service} does not pass the prod data image")
        if "iqa-ml" not in str(env.get("IQA_IMAGE_ML", "")):
            raise AssertionError(f"{airflow_service} does not pass the prod ml image")
        if "iqa-dvc-gate" not in str(env.get("IQA_IMAGE_DVC", "")):
            raise AssertionError(f"{airflow_service} does not pass the prod dvc gate image")
        if "IQA_IMAGE_TAG" not in str(env.get("IQA_IMAGE_DATA", "")):
            raise AssertionError(f"{airflow_service} data image is not tag-controlled")
        if "IQA_IMAGE_TAG" not in str(env.get("IQA_IMAGE_ML", "")):
            raise AssertionError(f"{airflow_service} ml image is not tag-controlled")
        if "IQA_IMAGE_TAG" not in str(env.get("IQA_IMAGE_DVC", "")):
            raise AssertionError(f"{airflow_service} dvc gate image is not tag-controlled")

    for term in EXPECTED_SMOKE_TERMS:
        if term not in smoke:
            raise AssertionError(f"smoke test does not cover: {term}")

    for term in [
        "docker compose -f docker-compose.yml -f docker-compose.prod.yml pull",
        "docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d",
        "Docker Hub",
        "IQA_PUBLISH_IMAGES",
        "IQA_IMAGE_REGISTRY",
        "IQA_IMAGE_TAG=sha-<commit>",
        "DOCKERHUB_USERNAME",
        "DOCKERHUB_TOKEN",
        "gh variable set",
        "gh secret set",
        "docker login",
        "Jamais de `latest`",
        "Kong",
        "Nginx",
        "MLflow Registry",
    ]:
        if term not in runbook + phase3_doc:
            raise AssertionError(f"deployment evidence docs miss: {term}")

    return {
        "published_services": sorted(PUBLISHED_IMAGE_SERVICES),
        "image_tag_source": "IQA_IMAGE_TAG",
        "registry_source": "IQA_IMAGE_REGISTRY",
        "recommended_tag_strategy": "ci_sha",
        "smoke_terms": EXPECTED_SMOKE_TERMS,
        "status": "validated",
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise AssertionError(f"YAML root is not a mapping: {path}")
    return payload


def _services(payload: dict[str, Any]) -> dict[str, Any]:
    services = payload.get("services")
    if not isinstance(services, dict):
        raise AssertionError("compose file has no services mapping")
    return services


if __name__ == "__main__":
    main()
