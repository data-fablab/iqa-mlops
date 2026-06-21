"""Tests for the container-task operator factory (issue 05, ADR 0008)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from iqa.dags import build_container_dag, data_image, dvc_image, make_container_task, ml_image
from iqa.dags.operators import _normalise_command, _task_environment

DAG_FOLDER = Path(__file__).parents[2] / "airflow" / "dags"
sys.path.insert(0, str(DAG_FOLDER))

def _has_docker_provider() -> bool:
    try:
        return importlib.util.find_spec("airflow.providers.docker.operators.docker") is not None
    except ModuleNotFoundError:
        return False


def _has_airflow_dag() -> bool:
    try:
        from airflow import DAG  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_DOCKER_PROVIDER = _has_docker_provider()
_HAS_AIRFLOW_DAG = _has_airflow_dag()


@pytest.mark.unit
def test_data_image_defaults_and_honours_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IQA_IMAGE_DATA", raising=False)
    assert data_image() == "iqa-data:local"
    monkeypatch.setenv("IQA_IMAGE_DATA", "registry/iqa-data:1.2.3")
    assert data_image() == "registry/iqa-data:1.2.3"


@pytest.mark.unit
def test_ml_image_defaults_and_honours_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IQA_IMAGE_ML", raising=False)
    assert ml_image() == "iqa-ml:local"
    monkeypatch.setenv("IQA_IMAGE_ML", "registry/iqa-ml:1.2.3")
    assert ml_image() == "registry/iqa-ml:1.2.3"


@pytest.mark.unit
def test_dvc_image_defaults_and_honours_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IQA_IMAGE_DVC", raising=False)
    assert dvc_image() == "iqa-dvc-gate:local"
    monkeypatch.setenv("IQA_IMAGE_DVC", "registry/iqa-dvc-gate:1.2.3")
    assert dvc_image() == "registry/iqa-dvc-gate:1.2.3"


@pytest.mark.unit
@pytest.mark.skipif(_HAS_AIRFLOW_DAG, reason="Airflow installed: the None-resilience path is unreachable")
def test_build_container_dag_returns_none_without_airflow() -> None:
    """The DAG module stays importable when Airflow is absent (CI): None, not raise."""
    called = False

    def _define() -> None:
        nonlocal called
        called = True

    dag = build_container_dag(
        dag_id="iqa_x",
        define=_define,
        params={"image": "iqa-data:local"},
        tags=["iqa"],
    )

    assert dag is None
    assert called is False  # never entered the DAG context


@pytest.mark.docker_contract
@pytest.mark.skipif(not _HAS_AIRFLOW_DAG, reason="Airflow not installed")
def test_build_container_dag_runs_define_inside_the_dag(monkeypatch: pytest.MonkeyPatch) -> None:
    """With Airflow available, define() runs inside the DAG and its id/params stick."""
    seen: dict[str, object] = {}

    def _define() -> None:
        from airflow.operators.empty import EmptyOperator

        EmptyOperator(task_id="noop")
        seen["ran"] = True

    dag = build_container_dag(
        dag_id="iqa_build_probe",
        define=_define,
        params={"image": "iqa-data:local"},
        tags=["iqa", "probe"],
    )

    assert seen.get("ran") is True
    assert dag is not None
    assert dag.dag_id == "iqa_build_probe"
    assert {t.task_id for t in dag.tasks} == {"noop"}


@pytest.mark.unit
def test_normalise_command_splits_strings_and_copies_lists() -> None:
    assert _normalise_command("iqa-api --reload") == ["iqa-api", "--reload"]
    assert _normalise_command(["iqa-api", "--reload"]) == ["iqa-api", "--reload"]
    assert _normalise_command(None) is None


@pytest.mark.unit
def test_task_environment_forwards_allowlisted_vars_and_explicit_env_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IQA_TASK_ENV_PASSTHROUGH", raising=False)
    monkeypatch.setenv("IQA_METADATA_DB_URL", "postgresql://iqa@postgres/iqa_metadata")
    monkeypatch.setenv("IQA_S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)  # unset -> not forwarded

    env = _task_environment({"IQA_SCENARIO": "x", "IQA_METADATA_DB_URL": "override"})

    assert env["IQA_S3_ENDPOINT_URL"] == "http://minio:9000"  # allowlisted, forwarded
    assert env["IQA_SCENARIO"] == "x"  # explicit per-task env
    assert env["IQA_METADATA_DB_URL"] == "override"  # explicit wins over allowlist
    assert "IQA_SERVICE_TOKEN" not in env  # allowlisted but unset -> omitted


@pytest.mark.unit
def test_task_environment_does_not_bulk_forward_the_scheduler_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only allowlisted names cross into task containers; nothing else leaks."""
    monkeypatch.delenv("IQA_TASK_ENV_PASSTHROUGH", raising=False)
    monkeypatch.setenv("SOME_SCHEDULER_SECRET", "do-not-leak")
    assert "SOME_SCHEDULER_SECRET" not in _task_environment(None)


@pytest.mark.unit
def test_task_environment_honours_the_passthrough_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IQA_TASK_ENV_PASSTHROUGH", "MY_EXTRA, ANOTHER_ONE")
    monkeypatch.setenv("MY_EXTRA", "v")
    monkeypatch.delenv("ANOTHER_ONE", raising=False)
    env = _task_environment(None)
    assert env["MY_EXTRA"] == "v"
    assert "ANOTHER_ONE" not in env  # listed but unset -> omitted


@pytest.mark.unit
def test_make_container_task_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_AIRFLOW_BACKEND", "nomad")
    with pytest.raises(ValueError, match="nomad"):
        make_container_task(task_id="t", image="iqa-data:local", command="iqa-run-ingestion")


@pytest.mark.unit
def test_factory_module_does_not_import_iqa_runtime() -> None:
    """ADR 0008: the factory must not drag the metier runtime into Airflow."""
    source = (Path(__file__).parents[2] / "src" / "iqa" / "dags" / "operators.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("import torch", "import pandas", "from iqa.api", "from iqa.inference"):
        assert forbidden not in source, f"factory must not contain '{forbidden}'"


@pytest.mark.docker_contract
@pytest.mark.skipif(not _HAS_DOCKER_PROVIDER, reason="apache-airflow-providers-docker not installed")
def test_make_container_task_builds_docker_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    from airflow.providers.docker.operators.docker import DockerOperator

    monkeypatch.setenv("IQA_AIRFLOW_BACKEND", "docker")
    monkeypatch.setenv("IQA_DOCKER_NETWORK", "iqa_net")
    op = make_container_task(
        task_id="run_ingestion",
        image="iqa-data:local",
        command="iqa-run-ingestion",
        env={"IQA_SCENARIO": "x"},
        pool="iqa_gpu",
    )
    assert isinstance(op, DockerOperator)
    assert op.task_id == "run_ingestion"
    assert op.image == "iqa-data:local"
    # explicit per-task env is always present; allowlisted vars may also be
    # forwarded from the ambient environment, so assert membership, not equality.
    assert op.environment["IQA_SCENARIO"] == "x"
    assert op.pool == "iqa_gpu"
    assert op.network_mode == "iqa_net"


@pytest.mark.docker_contract
@pytest.mark.skipif(not _HAS_DOCKER_PROVIDER, reason="apache-airflow-providers-docker not installed")
def test_make_container_task_can_mount_repo_and_gpu_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifecycle-style tasks need host repo data/cache plus the shared GPU lock."""
    monkeypatch.setenv("IQA_AIRFLOW_BACKEND", "docker")
    monkeypatch.setenv("IQA_GPU_LOCK_VOLUME", "iqa_gpu_lock")
    monkeypatch.setenv("IQA_GPU_LOCK_PATH", "/var/run/iqa-gpu/gpu.lock")
    monkeypatch.setenv("IQA_AIRFLOW_REPO_MOUNT_SOURCE", "/opt/iqa/iqa-mlops")
    monkeypatch.setenv("IQA_AIRFLOW_REPO_MOUNT_TARGET", "/opt/iqa/iqa-mlops")

    op = make_container_task(
        task_id="run_application_lifecycle",
        image="iqa-ml:local",
        command="iqa-run-replay-lifecycle-cycle",
        gpu_lock=True,
        repo_mount=True,
        working_dir="/opt/iqa/iqa-mlops",
    )

    mounts = {mount["Target"]: mount for mount in op.mounts}
    assert mounts["/var/run/iqa-gpu"]["Type"] == "volume"
    assert mounts["/var/run/iqa-gpu"]["Source"] == "iqa_gpu_lock"
    assert mounts["/opt/iqa/iqa-mlops"]["Type"] == "bind"
    assert mounts["/opt/iqa/iqa-mlops"]["Source"] == "/opt/iqa/iqa-mlops"
    assert op.environment["IQA_REPO_ROOT"] == "/opt/iqa/iqa-mlops"
    assert op.working_dir == "/opt/iqa/iqa-mlops"


@pytest.mark.docker_contract
@pytest.mark.skipif(not _HAS_DOCKER_PROVIDER, reason="apache-airflow-providers-docker not installed")
def test_tracer_dag_builds_one_container_task() -> None:
    import iqa_container_tracer

    dag = iqa_container_tracer.dag
    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {"run_container"}
