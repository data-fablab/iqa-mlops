"""Tests for the container-task operator factory (issue 05, ADR 0008)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from iqa.dags.operators import _normalise_command, make_container_task

DAG_FOLDER = Path(__file__).parents[2] / "airflow" / "dags"
sys.path.insert(0, str(DAG_FOLDER))

def _has_docker_provider() -> bool:
    try:
        return importlib.util.find_spec("airflow.providers.docker.operators.docker") is not None
    except ModuleNotFoundError:
        return False


_HAS_DOCKER_PROVIDER = _has_docker_provider()


@pytest.mark.unit
def test_normalise_command_splits_strings_and_copies_lists() -> None:
    assert _normalise_command("iqa-api --reload") == ["iqa-api", "--reload"]
    assert _normalise_command(["iqa-api", "--reload"]) == ["iqa-api", "--reload"]
    assert _normalise_command(None) is None


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
    assert op.environment == {"IQA_SCENARIO": "x"}
    assert op.pool == "iqa_gpu"
    assert op.network_mode == "iqa_net"


@pytest.mark.docker_contract
@pytest.mark.skipif(not _HAS_DOCKER_PROVIDER, reason="apache-airflow-providers-docker not installed")
def test_tracer_dag_builds_one_container_task() -> None:
    import iqa_container_tracer

    dag = iqa_container_tracer.dag
    assert dag is not None
    assert {t.task_id for t in dag.tasks} == {"run_container"}
