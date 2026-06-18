"""Contract tests for Airflow DAGs (IQA1_KEN09)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


# Add airflow/dags to path for imports (repo_root/airflow/dags)
DAG_FOLDER = Path(__file__).parents[2] / "airflow" / "dags"
sys.path.insert(0, str(DAG_FOLDER))


def _read_dag_source(name: str = "iqa_lifecycle.py") -> str:
    """Read an Airflow DAG source code file."""
    return (DAG_FOLDER / name).read_text(encoding="utf-8")


@pytest.mark.docker_contract
def test_iqa_lifecycle_dag_imports_without_error() -> None:
    """Test that the iqa_lifecycle DAG can be imported."""
    try:
        import iqa_lifecycle  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")


@pytest.mark.docker_contract
def test_iqa_lifecycle_dag_has_eight_tasks() -> None:
    """Test that iqa_lifecycle DAG has all 8 pipeline tasks."""
    try:
        import iqa_lifecycle
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag = iqa_lifecycle.dag
    if dag is None:
        pytest.skip("DAG is None (Airflow not available)")

    task_ids = {task.task_id for task in dag.tasks}
    expected_tasks = {"lifecycle_decision", "dataset", "train", "eval", "gates", "mlflow", "promotion", "reload"}

    assert expected_tasks <= task_ids, f"Missing tasks: {expected_tasks - task_ids}"


@pytest.mark.unit
def test_iqa_lifecycle_dag_source_declares_eight_tasks() -> None:
    """Test that DAG source declares all 8 tasks (code-level check)."""
    source = _read_dag_source()
    task_ids = ["lifecycle_decision", "dataset", "train", "eval", "gates", "mlflow", "promotion", "reload"]

    for task_id in task_ids:
        assert f'task_id="{task_id}"' in source, f"Task {task_id} not declared in DAG source"


@pytest.mark.unit
def test_iqa_lifecycle_dag_source_declares_dependencies() -> None:
    """Test that DAG source declares linear dependencies (code-level check)."""
    source = _read_dag_source()

    # Check for the dependency chain pattern: lifecycle_decision >> dataset >> train >> eval >> gates >> mlflow >> promotion >> reload
    # Support both old (task_*) and new (op_*) variable naming
    assert (
        "op_lifecycle_decision >> op_dataset" in source
        or "lifecycle_decision >> dataset" in source
        or "task_lifecycle_decision >> task_dataset" in source
    )
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


@pytest.mark.unit
def test_ingestion_dag_runs_data_image_via_factory() -> None:
    """Ingestion DAG (issue 07) launches the data container, not a local CLI."""
    ingestion = _read_dag_source("iqa_ingestion.py")

    assert "BashOperator(" not in ingestion
    assert "make_container_task(" in ingestion
    # Templated params passed as argv elements (no shell quoting).
    assert '"iqa-run-ingestion"' in ingestion
    assert '"{{ params.manifest }}"' in ingestion
    assert '"{{ params.source }}"' in ingestion
    assert '"{{ params.scenario_id }}"' in ingestion


@pytest.mark.unit
def test_replay_dag_containerises_via_factory() -> None:
    """Replay DAG (issue 12) runs iqa-run-replay as a data-image container.

    The BashOperator is replaced by make_container_task with templated argv
    elements (no shell, no quoting); the DAG no longer references iqa metier code.
    """
    replay = _read_dag_source("iqa_replay.py")

    assert "make_container_task(" in replay
    assert '"iqa-run-replay"' in replay
    assert '"{{ params.scenario_id }}"' in replay
    assert '"{{ params.plan }}"' in replay
    # No BashOperator shell form left.
    assert "BashOperator(" not in replay
    assert "bash_command" not in replay


@pytest.mark.unit
def test_monitoring_dag_containerises_via_factory() -> None:
    """Monitoring DAG (issue 13) runs iqa-run-monitoring as a data-image container.

    The BashOperator is replaced by make_container_task with templated argv
    elements; drift_confirmed is passed as a value (not a Jinja-conditional flag)
    and the thresholds config is evaluated in-container.
    """
    monitoring = _read_dag_source("iqa_monitoring.py")

    assert "make_container_task(" in monitoring
    assert '"iqa-run-monitoring"' in monitoring
    assert '"{{ params.conforming_validated_count }}"' in monitoring
    assert '"--drift-confirmed", "{{ params.drift_confirmed }}"' in monitoring
    assert '"{{ params.roi_fail_rate }}"' in monitoring
    assert '"{{ params.thresholds_config }}"' in monitoring
    # No BashOperator shell form left.
    assert "BashOperator(" not in monitoring
    assert "bash_command" not in monitoring


@pytest.mark.unit
def test_lifecycle_dag_containerises_decision_and_dataset_via_factory() -> None:
    """Lifecycle DAG (issue 08) runs lifecycle_decision + dataset as containers.

    The two leading stages move to the data image via the operator factory
    (issue 11 later containerises the whole tail, removing every PythonOperator).
    """
    source = _read_dag_source("iqa_lifecycle.py")

    assert "make_container_task(" in source
    # Decision + dataset call the data-image boundary scripts with templated argv.
    assert '"iqa-run-lifecycle-decision"' in source
    assert '"iqa-run-dataset"' in source
    assert '"{{ params.scenario_id }}"' in source
    assert '"{{ params.manifest }}"' in source
    # These two tasks no longer route through the iqa lifecycle_tasks callables.
    assert "task_lifecycle_decision" not in source
    assert "task_dataset" not in source


@pytest.mark.unit
def test_lifecycle_dag_containerises_train_and_eval_on_ml_image_with_gpu_lock() -> None:
    """Lifecycle DAG (issue 09) runs train + eval as GPU-locked ml containers.

    train/eval move to the ml image via the factory with the iqa_gpu pool and the
    shared single-GPU lock; the tail (gates..reload) stays on PythonOperator.
    """
    source = _read_dag_source("iqa_lifecycle.py")

    assert '"iqa-run-train"' in source
    assert '"iqa-run-eval"' in source
    assert '"{{ params.ml_image }}"' in source
    # GPU-bound: factory mounts the shared lock and the iqa_gpu pool (slots=1).
    assert "gpu_lock=True" in source
    assert "pool=GPU_POOL" in source
    # These two tasks no longer route through the iqa lifecycle_tasks callables.
    assert "task_train" not in source
    assert "task_eval" not in source


@pytest.mark.unit
def test_lifecycle_dag_containerises_gates_and_mlflow_via_factory() -> None:
    """Lifecycle DAG (issue 10) runs gates + mlflow as containers.

    gates evaluates promotion_gates.yaml on the data image (blocks on failure);
    mlflow resolves the scenario-isolated name on the ml image. Issue 11 then
    containerises the promotion/reload tail.
    """
    source = _read_dag_source("iqa_lifecycle.py")

    assert '"iqa-run-gates"' in source
    assert '"iqa-run-mlflow"' in source
    assert '"{{ params.gates_config }}"' in source
    # gates/mlflow no longer route through the iqa lifecycle_tasks callables.
    assert "task_gates" not in source
    assert "task_mlflow" not in source


@pytest.mark.unit
def test_lifecycle_dag_containerises_promotion_and_reload_via_factory() -> None:
    """Lifecycle DAG (issue 11) runs promotion + reload as containers.

    The last two stages move to the factory, so ADR 0008 is fully resolved: no
    PythonOperator and no iqa lifecycle_tasks import remain in the DAG. promotion
    runs on the ml image (MLflow is the source of truth); reload runs on the data
    image and only acts for prod promotions.
    """
    source = _read_dag_source("iqa_lifecycle.py")

    assert '"iqa-run-promotion"' in source
    assert '"iqa-run-reload"' in source
    assert '"{{ params.target_stage }}"' in source
    # No PythonOperator instantiation and no lifecycle_tasks callables anymore.
    assert "PythonOperator(" not in source
    assert "task_promotion" not in source
    assert "task_reload" not in source
    assert "lifecycle_tasks" not in source


@pytest.mark.unit
def test_dvc_reproducibility_dag_declares_safe_dvc_gate() -> None:
    """DVC is exposed to Airflow as an explicit reproducibility gate."""
    source = _read_dag_source("iqa_dvc_reproducibility.py")

    assert 'dag_id="iqa_dvc_reproducibility"' in source
    assert 'task_id="dvc_reproducibility_check"' in source
    assert '"with_network": False' in source
    # Image-friendly default: the git-diff regeneration check stays in CI.
    assert '"skip_regeneration": True' in source
    assert '"dvc_target": "data/raw/hss-iad.dvc"' in source
    assert "iqa-check-dvc-reproducibility" in source
    # Containerised via the factory on the dedicated dvc-gate image (ADR 0008):
    # booleans pass as templated values, no shell-conditional flags.
    assert "make_container_task" in source
    assert "dvc_image()" in source
    assert '"--with-network", "{{ params.with_network }}"' in source
    assert '"--skip-regeneration", "{{ params.skip_regeneration }}"' in source
    assert '"--dvc-target", "{{ params.dvc_target }}"' in source
    assert "{% if params.with_network %}" not in source  # no shell-conditional flags
    assert "dvc push" not in source


@pytest.mark.docker_contract
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
    expected_chain = ["lifecycle_decision", "dataset", "train", "eval", "gates", "mlflow", "promotion", "reload"]

    # Verify downstream dependencies
    for i in range(len(expected_chain) - 1):
        task = dag.get_task(expected_chain[i])
        next_task = dag.get_task(expected_chain[i + 1])
        downstream_ids = {t.task_id for t in task.downstream_list}
        assert (
            next_task.task_id in downstream_ids
        ), f"{expected_chain[i]} should have {expected_chain[i+1]} as downstream"


@pytest.mark.docker_contract
def test_iqa_lifecycle_dag_passes_dagbag_validation() -> None:
    """Test that iqa_lifecycle DAG passes Airflow DagBag validation."""
    try:
        from airflow.models import DagBag

        import iqa_lifecycle
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    if iqa_lifecycle.dag is None:
        pytest.skip("DAG is None (Docker provider not available)")

    dag_bag = DagBag(dag_folder=str(DAG_FOLDER), include_examples=False)

    assert "iqa_lifecycle" in dag_bag.dag_ids, "iqa_lifecycle DAG not found in DagBag"
    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"

    dag = dag_bag.get_dag("iqa_lifecycle")
    assert dag is not None, "iqa_lifecycle DAG is None"
    assert len(dag.tasks) == 8, f"Expected 8 tasks, got {len(dag.tasks)}"


@pytest.mark.unit
def test_lifecycle_trigger_dag_evaluates_in_container_and_triggers_lifecycle() -> None:
    """Trigger DAG (issue 16) evaluates the event rule in a container, then fires
    iqa_lifecycle via native Airflow glue -- no manual trigger, no iqa import.

    evaluate_decision runs iqa-run-lifecycle-decision on the data image; a
    ShortCircuitOperator gates on the container's trigger_lifecycle decision; a
    TriggerDagRunOperator launches iqa_lifecycle and relays the signal as conf.
    """
    trigger = _read_dag_source("iqa_lifecycle_trigger.py")

    # The metier decision runs in the data-image container (ADR 0008).
    assert "make_container_task(" in trigger
    assert '"iqa-run-lifecycle-decision"' in trigger
    assert '"{{ params.conforming_validated_count }}"' in trigger
    assert '"--drift-confirmed", "{{ params.drift_confirmed }}"' in trigger
    # Native Airflow glue: gate on the decision, then trigger the lifecycle.
    assert "ShortCircuitOperator(" in trigger
    assert "TriggerDagRunOperator(" in trigger
    assert 'trigger_dag_id=LIFECYCLE_DAG_ID' in trigger or 'trigger_dag_id="iqa_lifecycle"' in trigger
    assert "op_evaluate_decision >> op_gate_on_decision >> op_trigger_lifecycle" in trigger
    # No shell / no eager iqa import in the scheduler.
    assert "BashOperator(" not in trigger
    assert "bash_command" not in trigger


@pytest.mark.docker_contract
def test_lifecycle_trigger_dag_has_trigger_chain() -> None:
    """Trigger DAG wires evaluate_decision -> gate_on_decision -> trigger_lifecycle."""
    try:
        import iqa_lifecycle_trigger
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag = iqa_lifecycle_trigger.dag
    if dag is None:
        pytest.skip("DAG is None (Airflow provider not available)")

    expected_chain = ["evaluate_decision", "gate_on_decision", "trigger_lifecycle"]
    task_ids = {task.task_id for task in dag.tasks}
    assert set(expected_chain) <= task_ids, f"Missing tasks: {set(expected_chain) - task_ids}"

    for upstream, downstream in zip(expected_chain, expected_chain[1:]):
        task = dag.get_task(upstream)
        downstream_ids = {t.task_id for t in task.downstream_list}
        assert downstream in downstream_ids, f"{upstream} should have {downstream} downstream"
