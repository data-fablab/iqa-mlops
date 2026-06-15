"""Contract tests for Airflow DAGs (IQA1_KEN09)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest


# Add airflow/dags to path for imports
DAG_FOLDER = Path(__file__).parent.parent / "airflow" / "dags"
sys.path.insert(0, str(DAG_FOLDER))


def _read_dag_source() -> str:
    """Read the iqa_lifecycle DAG source code."""
    return (DAG_FOLDER / "iqa_lifecycle.py").read_text(encoding="utf-8")


def test_iqa_lifecycle_dag_imports_without_error() -> None:
    """Test that the iqa_lifecycle DAG can be imported."""
    try:
        import iqa_lifecycle  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")


def test_iqa_lifecycle_dag_has_seven_tasks() -> None:
    """Test that iqa_lifecycle DAG has all 7 pipeline tasks."""
    try:
        import iqa_lifecycle
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag = iqa_lifecycle.dag
    if dag is None:
        pytest.skip("DAG is None (Airflow not available)")

    task_ids = {task.task_id for task in dag.tasks}
    expected_tasks = {"dataset", "train", "eval", "gates", "mlflow", "promotion", "reload"}

    assert expected_tasks <= task_ids, f"Missing tasks: {expected_tasks - task_ids}"


def test_iqa_lifecycle_dag_source_declares_seven_tasks() -> None:
    """Test that DAG source declares all 7 tasks (code-level check)."""
    source = _read_dag_source()
    task_ids = ["dataset", "train", "eval", "gates", "mlflow", "promotion", "reload"]

    for task_id in task_ids:
        assert f'task_id="{task_id}"' in source, f"Task {task_id} not declared in DAG source"


def test_iqa_lifecycle_dag_source_declares_dependencies() -> None:
    """Test that DAG source declares linear dependencies (code-level check)."""
    source = _read_dag_source()

    # Check for the dependency chain pattern: dataset >> train >> eval >> gates >> mlflow >> promotion >> reload
    # Support both old (task_*) and new (op_*) variable naming
    assert (
        ">> op_train >>" in source
        or "op_dataset >> op_train" in source
        or ">> task_train >>" in source
        or "task_dataset >> task_train" in source
    )
    assert (
        ">> op_eval >>" in source
        or "op_train >> op_eval" in source
        or ">> task_eval >>" in source
        or "task_train >> task_eval" in source
    )
    assert (
        ">> op_gates >>" in source
        or "op_eval >> op_gates" in source
        or ">> task_gates >>" in source
        or "task_eval >> task_gates" in source
    )
    assert (
        ">> op_mlflow >>" in source
        or "op_gates >> op_mlflow" in source
        or ">> task_mlflow >>" in source
        or "task_gates >> task_mlflow" in source
    )
    assert (
        ">> op_promotion >>" in source
        or "op_mlflow >> op_promotion" in source
        or ">> task_promotion >>" in source
        or "task_mlflow >> task_promotion" in source
    )
    assert (
        ">> op_reload" in source
        or "op_promotion >> op_reload" in source
        or ">> task_reload" in source
        or "task_promotion >> task_reload" in source
    )


def test_iqa_lifecycle_dag_has_linear_dependencies() -> None:
    """Test that iqa_lifecycle DAG has linear dependencies: dataset→train→eval→gates→mlflow→promotion→reload."""
    try:
        import iqa_lifecycle
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag = iqa_lifecycle.dag
    if dag is None:
        pytest.skip("DAG is None (Airflow not available)")

    # Expected linear chain
    expected_chain = ["dataset", "train", "eval", "gates", "mlflow", "promotion", "reload"]

    # Verify downstream dependencies
    for i in range(len(expected_chain) - 1):
        task = dag.get_task(expected_chain[i])
        next_task = dag.get_task(expected_chain[i + 1])
        downstream_ids = {t.task_id for t in task.downstream_list}
        assert (
            next_task.task_id in downstream_ids
        ), f"{expected_chain[i]} should have {expected_chain[i+1]} as downstream"


def test_iqa_lifecycle_dag_passes_dagbag_validation() -> None:
    """Test that iqa_lifecycle DAG passes Airflow DagBag validation."""
    try:
        from airflow.models import DagBag
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag_bag = DagBag(dag_folder=str(DAG_FOLDER), include_examples=False)

    assert "iqa_lifecycle" in dag_bag.dag_ids, "iqa_lifecycle DAG not found in DagBag"
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"

    dag = dag_bag.get_dag("iqa_lifecycle")
    assert dag is not None, "iqa_lifecycle DAG is None"
    assert len(dag.tasks) == 7, f"Expected 7 tasks, got {len(dag.tasks)}"
