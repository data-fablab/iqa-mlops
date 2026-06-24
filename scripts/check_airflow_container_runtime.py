"""Validate Phase 3 Airflow container-runtime evidence without starting Airflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DAGS_DIR = ROOT / "airflow" / "dags"
COMPOSE_FILE = ROOT / "deploy" / "docker-compose.yml"
OPERATORS_FILE = ROOT / "src" / "iqa" / "dags" / "operators.py"
AIRFLOW_DOC = ROOT / "docs" / "airflow-container-runtime-evidence.md"
DEPLOY_RUNBOOK = ROOT / "docs" / "deploy_runbook.md"

CONTAINER_DAGS = {
    "iqa_ingestion.py": "iqa-run-ingestion",
    "iqa_replay.py": "iqa-run-replay",
    "iqa_monitoring.py": "iqa-run-monitoring",
    "iqa_lifecycle.py": "iqa-run-replay-lifecycle-cycle",
    "iqa_lifecycle_trigger.py": "iqa-run-lifecycle-decision",
}
EXPECTED_DAG_IDS = {
    "iqa_ingestion",
    "iqa_replay",
    "iqa_monitoring",
    "iqa_lifecycle",
    "iqa_lifecycle_trigger",
    "iqa_dvc_reproducibility",
}
FORBIDDEN_RUNTIME_IMPORTS = (
    "from iqa.api",
    "from iqa.inference",
    "from iqa.training",
    "import torch",
    "import pandas",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable evidence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = build_airflow_container_runtime_evidence()
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("Airflow container-runtime evidence checks passed.")


def build_airflow_container_runtime_evidence() -> dict[str, Any]:
    compose = _read_yaml(COMPOSE_FILE)
    services = _services(compose)
    scheduler = _service(services, "airflow-scheduler")
    webserver = _service(services, "airflow-webserver")
    volumes = compose.get("volumes", {})
    networks = compose.get("networks", {})

    dag_sources = {path.name: path.read_text(encoding="utf-8") for path in DAGS_DIR.glob("iqa_*.py")}
    missing = [dag_id for dag_id in EXPECTED_DAG_IDS if f'dag_id="{dag_id}"' not in "\n".join(dag_sources.values())]
    if missing:
        raise AssertionError(f"missing expected Airflow DAG ids: {missing}")

    for filename, command in CONTAINER_DAGS.items():
        source = dag_sources.get(filename)
        if source is None:
            raise AssertionError(f"missing DAG source: {filename}")
        if "make_container_task(" not in source:
            raise AssertionError(f"{filename} does not use make_container_task")
        if command not in source:
            raise AssertionError(f"{filename} does not call expected command: {command}")
        if "except ImportError" in source or "build_container_dag is not None" in source:
            raise AssertionError(f"{filename} hides missing iqa.dags imports instead of surfacing a broken DAG")
        if "PythonOperator(" in source:
            raise AssertionError(f"{filename} still instantiates PythonOperator")
        for forbidden in FORBIDDEN_RUNTIME_IMPORTS:
            if forbidden in source:
                raise AssertionError(f"{filename} imports forbidden runtime dependency: {forbidden}")

    dvc_source = dag_sources.get("iqa_dvc_reproducibility.py", "")
    if "iqa-check-dvc-reproducibility" not in dvc_source:
        raise AssertionError("DVC reproducibility DAG does not call iqa-check-dvc-reproducibility")
    if "make_container_task(" not in dvc_source or "BashOperator(" in dvc_source:
        raise AssertionError("DVC reproducibility DAG must run as a data container task")
    if "dvc push" in dvc_source:
        raise AssertionError("DVC reproducibility DAG must not call dvc push directly")
    if '"with_network"' not in dvc_source or '"skip_regeneration"' not in dvc_source:
        raise AssertionError("DVC reproducibility DAG misses explicit runtime params")

    operators = OPERATORS_FILE.read_text(encoding="utf-8")
    for term in [
        "DockerOperator",
        "IQA_AIRFLOW_BACKEND",
        "IQA_DOCKER_NETWORK",
        "DEFAULT_TASK_ENV_PASSTHROUGH",
        "IQA_AIRFLOW_REPO_MOUNT_SOURCE",
        "repo_mount",
        "DeviceRequest",
    ]:
        if term not in operators:
            raise AssertionError(f"operator factory misses: {term}")

    scheduler_env = scheduler.get("environment", {})
    scheduler_volumes = scheduler.get("volumes", [])
    webserver_volumes = webserver.get("volumes", [])
    if scheduler_env.get("IQA_AIRFLOW_BACKEND") != "docker":
        raise AssertionError("airflow-scheduler does not default to docker backend")
    if scheduler_env.get("IQA_DOCKER_NETWORK") != "iqa_net":
        raise AssertionError("airflow-scheduler does not join task containers to iqa_net")
    if scheduler_env.get("IQA_GPU_LOCK_VOLUME") != "iqa_gpu_lock":
        raise AssertionError("airflow-scheduler does not expose the shared GPU lock volume")
    if (
        "IQA_IMAGE_DATA" not in scheduler_env
        or "IQA_IMAGE_ML" not in scheduler_env
        or "IQA_IMAGE_DVC" not in scheduler_env
    ):
        raise AssertionError("airflow-scheduler does not expose task runtime images")
    webserver_env = webserver.get("environment", {})
    if (
        "IQA_IMAGE_DATA" not in webserver_env
        or "IQA_IMAGE_ML" not in webserver_env
        or "IQA_IMAGE_DVC" not in webserver_env
    ):
        raise AssertionError("airflow-webserver does not expose task runtime images")
    if not _volume_present(scheduler_volumes, "/var/run/docker.sock:/var/run/docker.sock"):
        raise AssertionError("airflow-scheduler does not mount the Docker socket")
    if not _volume_present(scheduler_volumes, "../src:/opt/iqa/src:ro"):
        raise AssertionError("airflow-scheduler does not mount the lightweight iqa.dags factory")
    if not _volume_present(webserver_volumes, "../src:/opt/iqa/src:ro"):
        raise AssertionError("airflow-webserver does not mount the lightweight iqa.dags factory")
    if volumes.get("gpu_lock", {}).get("name") != "iqa_gpu_lock":
        raise AssertionError("compose does not define stable iqa_gpu_lock volume")
    if networks.get("default", {}).get("name") != "iqa_net":
        raise AssertionError("compose does not define stable iqa_net network")

    lifecycle = dag_sources["iqa_lifecycle.py"]
    for term in [
        "run_application_lifecycle",
        "--mode",
        "progressive-train",
        "--max-cycles",
        "--lifecycle-interval",
        "--promotion-min-delta",
        "--anchor-good-manifest",
        "--reference-eval-manifest",
        "--max-good-red-regression",
        "--candidate-init-policy",
        "--publish-minio",
        "--wait-for-gpu",
        "max_active_runs=1",
        "execution_timeout",
        "retries=0",
    ]:
        if term not in lifecycle:
            raise AssertionError(f"application lifecycle DAG misses: {term}")
    for legacy_command in ["iqa-run-train", "iqa-run-eval", "iqa-run-gates", "iqa-run-promotion"]:
        if legacy_command in lifecycle:
            raise AssertionError(f"application lifecycle DAG still calls legacy command: {legacy_command}")
    if 'pool=GPU_POOL' not in lifecycle or 'gpu_lock=True' not in lifecycle:
        raise AssertionError("application lifecycle task is not protected by GPU pool and lock")
    for term in ['repo_mount=True', 'working_dir="/opt/iqa/iqa-mlops"', '"repo_root": "/opt/iqa/iqa-mlops"']:
        if term not in lifecycle:
            raise AssertionError(f"application lifecycle DAG misses workspace mount contract: {term}")

    docs = AIRFLOW_DOC.read_text(encoding="utf-8") + "\n" + DEPLOY_RUNBOOK.read_text(encoding="utf-8")
    for term in [
        "iqa-check-airflow-container-runtime --json",
        "airflow dags list",
        "airflow dags list-import-errors",
        "airflow dags unpause iqa_dvc_reproducibility",
        "airflow dags unpause iqa_lifecycle_trigger",
        "airflow dags trigger iqa_dvc_reproducibility",
        "airflow dags trigger iqa_lifecycle_trigger",
        "iqa-run-replay-lifecycle-cycle",
        "pipeline applicatif Feature-AE",
        "Docker Compose orchestre les services longs",
        "/var/run/docker.sock",
        "Kubernetes reste Phase 4",
        "pas de training via CI",
    ]:
        if term not in docs:
            raise AssertionError(f"Airflow runtime evidence docs miss: {term}")

    return {
        "backend": "docker",
        "container_dags": sorted(name.removesuffix(".py") for name in CONTAINER_DAGS),
        "dvc_gate": "iqa_dvc_reproducibility",
        "gpu_pool": "iqa_gpu",
        "lifecycle_command": "iqa-run-replay-lifecycle-cycle",
        "lifecycle_mode": "progressive-train",
        "network": "iqa_net",
        "promotion_policy": "candidate_must_pass_reference_guardrail_and_progressive_factory_panel",
        "registry_stage": "test",
        "gpu_device_request": "all",
        "repo_mount": "/opt/iqa/iqa-mlops",
        "server_commands": [
            "airflow dags list",
            "airflow dags list-import-errors",
            "airflow pools list",
            "airflow dags unpause iqa_dvc_reproducibility",
            "airflow dags unpause iqa_lifecycle_trigger",
            "airflow dags trigger iqa_dvc_reproducibility",
            "airflow dags trigger iqa_lifecycle_trigger",
        ],
        "socket_mount": "/var/run/docker.sock",
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


def _service(services: dict[str, Any], name: str) -> dict[str, Any]:
    service = services.get(name)
    if not isinstance(service, dict):
        raise AssertionError(f"missing compose service: {name}")
    return service


def _volume_present(volumes: list[Any], expected: str) -> bool:
    return any(str(volume) == expected for volume in volumes)


if __name__ == "__main__":
    main()
