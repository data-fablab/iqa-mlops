from pathlib import Path

import yaml

from scripts.check_airflow_container_runtime import build_airflow_container_runtime_evidence

ROOT = Path(".")


def test_airflow_container_runtime_evidence_command_is_public() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        'iqa-check-airflow-container-runtime = "scripts.check_airflow_container_runtime:main"'
        in pyproject
    )


def test_airflow_container_runtime_static_checks_pass() -> None:
    evidence = build_airflow_container_runtime_evidence()

    assert evidence["status"] == "validated"
    assert evidence["backend"] == "docker"
    assert evidence["dvc_gate"] == "iqa_dvc_reproducibility"
    assert evidence["gpu_pool"] == "iqa_gpu"
    assert evidence["lifecycle_command"] == "iqa-run-replay-lifecycle-cycle"
    assert evidence["lifecycle_mode"] == "progressive-train"
    assert evidence["network"] == "iqa_net"
    assert evidence["promotion_policy"] == "candidate_must_pass_reference_guardrail_and_progressive_factory_panel"
    assert evidence["registry_stage"] == "test"
    assert evidence["gpu_device_request"] == "all"
    assert set(evidence["container_dags"]) >= {
        "iqa_ingestion",
        "iqa_replay",
        "iqa_monitoring",
        "iqa_lifecycle",
        "iqa_lifecycle_trigger",
    }


def test_business_airflow_dags_use_container_factory_without_runtime_imports() -> None:
    dag_dir = ROOT / "airflow" / "dags"
    business_dags = [
        "iqa_ingestion.py",
        "iqa_replay.py",
        "iqa_monitoring.py",
        "iqa_lifecycle.py",
        "iqa_lifecycle_trigger.py",
    ]

    for dag_name in business_dags:
        source = (dag_dir / dag_name).read_text(encoding="utf-8")
        assert "make_container_task(" in source
        assert "except ImportError" not in source
        assert "build_container_dag is not None" not in source
        assert "PythonOperator(" not in source
        for forbidden in ["from iqa.api", "from iqa.inference", "from iqa.training", "import torch"]:
            assert forbidden not in source


def test_dvc_reproducibility_dag_stays_explicit_and_does_not_push() -> None:
    source = (ROOT / "airflow" / "dags" / "iqa_dvc_reproducibility.py").read_text(encoding="utf-8")

    assert 'dag_id="iqa_dvc_reproducibility"' in source
    assert "make_container_task(" in source
    assert "iqa-check-dvc-reproducibility" in source
    assert '"with_network": False' in source
    assert '"skip_regeneration": True' in source
    assert '"--dvc-target", "{{ params.dvc_target }}"' in source
    assert "BashOperator(" not in source
    assert "dvc push" not in source


def test_airflow_scheduler_compose_exposes_docker_backend_network_and_gpu_lock() -> None:
    compose = yaml.safe_load((ROOT / "deploy" / "docker-compose.yml").read_text(encoding="utf-8"))
    scheduler = compose["services"]["airflow-scheduler"]
    webserver = compose["services"]["airflow-webserver"]

    assert scheduler["environment"]["IQA_AIRFLOW_BACKEND"] == "docker"
    assert scheduler["environment"]["IQA_DOCKER_NETWORK"] == "iqa_net"
    assert scheduler["environment"]["IQA_GPU_LOCK_VOLUME"] == "iqa_gpu_lock"
    assert scheduler["environment"]["IQA_IMAGE_DATA"] == "${IQA_IMAGE_DATA:-iqa-data:local}"
    assert scheduler["environment"]["IQA_IMAGE_ML"] == "${IQA_IMAGE_ML:-iqa-ml:local}"
    assert scheduler["environment"]["IQA_IMAGE_DVC"] == "${IQA_IMAGE_DVC:-iqa-dvc-gate:local}"
    assert scheduler["environment"]["MLFLOW_TRACKING_URI"] == "${IQA_MLFLOW_TRACKING_URI:-http://mlflow:5000}"
    assert webserver["environment"]["IQA_IMAGE_DATA"] == "${IQA_IMAGE_DATA:-iqa-data:local}"
    assert webserver["environment"]["IQA_IMAGE_ML"] == "${IQA_IMAGE_ML:-iqa-ml:local}"
    assert webserver["environment"]["IQA_IMAGE_DVC"] == "${IQA_IMAGE_DVC:-iqa-dvc-gate:local}"
    assert webserver["environment"]["MLFLOW_TRACKING_URI"] == "${IQA_MLFLOW_TRACKING_URI:-http://mlflow:5000}"
    assert "/var/run/docker.sock:/var/run/docker.sock" in scheduler["volumes"]
    assert "../src:/opt/iqa/src:ro" in scheduler["volumes"]
    assert "../src:/opt/iqa/src:ro" in webserver["volumes"]
    assert compose["volumes"]["gpu_lock"]["name"] == "iqa_gpu_lock"
    assert compose["networks"]["default"]["name"] == "iqa_net"


def test_airflow_runtime_docs_cover_server_evidence_and_security_boundary() -> None:
    docs = "\n".join(
        [
            (ROOT / "docs" / "airflow-container-runtime-evidence.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "deploy_runbook.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "index.md").read_text(encoding="utf-8"),
        ]
    )

    for expected in [
        "iqa-check-airflow-container-runtime --json",
        "airflow dags list",
        "airflow dags list-import-errors",
        "airflow pools list",
        "airflow dags unpause iqa_dvc_reproducibility",
        "airflow dags unpause iqa_lifecycle_trigger",
        "airflow dags trigger iqa_dvc_reproducibility",
        "airflow dags trigger iqa_lifecycle_trigger",
        "airflow dags trigger iqa_lifecycle",
        "iqa-run-replay-lifecycle-cycle",
        "pipeline applicatif Feature-AE",
        "Docker Compose orchestre les services longs",
        "/var/run/docker.sock",
        "DockerOperator",
        "Kubernetes reste Phase 4",
        "pas de training via CI",
    ]:
        assert expected in docs
