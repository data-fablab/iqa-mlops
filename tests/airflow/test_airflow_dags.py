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
def test_iqa_lifecycle_dag_has_application_lifecycle_task() -> None:
    """Test that iqa_lifecycle DAG exposes the application lifecycle task."""
    try:
        import iqa_lifecycle
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag = iqa_lifecycle.dag
    if dag is None:
        pytest.skip("DAG is None (Airflow not available)")

    assert {task.task_id for task in dag.tasks} == {"run_application_lifecycle"}


@pytest.mark.unit
def test_iqa_lifecycle_dag_source_declares_application_lifecycle_task() -> None:
    """The DAG runs the application Feature-AE lifecycle, not the legacy split pipeline."""
    source = _read_dag_source()

    assert 'task_id="run_application_lifecycle"' in source
    assert "iqa-run-replay-lifecycle-cycle" in source
    for expected in [
        "--mode",
        "progressive-train",
        "--max-cycles",
        "--max-steps",
        "--gate-eval-profile",
        "--lifecycle-interval",
        "--promotion-min-delta",
        "--publish-minio",
        "--wait-for-gpu",
    ]:
        assert expected in source


@pytest.mark.unit
def test_iqa_lifecycle_dag_source_does_not_call_legacy_lifecycle_commands() -> None:
    """Airflow no longer duplicates the internal lifecycle chain as separate tasks."""
    source = _read_dag_source()

    for legacy in ["iqa-run-train", "iqa-run-eval", "iqa-run-gates", "iqa-run-promotion", "iqa-run-reload"]:
        assert legacy not in source


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
def test_lifecycle_dag_runs_reference_application_pipeline_via_factory() -> None:
    """Lifecycle DAG runs the reference Feature-AE application pipeline container."""
    source = _read_dag_source("iqa_lifecycle.py")

    assert "make_container_task(" in source
    assert "iqa-run-replay-lifecycle-cycle" in source
    assert "{{ params.scenario_id }}" in source
    assert "{{ params.image_root }}" in source
    assert "{{ params.mode }}" in source
    assert "pipeline" in source.lower()


@pytest.mark.unit
def test_lifecycle_dag_runs_on_ml_image_with_gpu_lock() -> None:
    """Application lifecycle runs on the ml image with the GPU pool and lock."""
    source = _read_dag_source("iqa_lifecycle.py")

    assert '"{{ params.ml_image }}"' in source
    assert "gpu_lock=True" in source
    assert "repo_mount=True" in source
    assert 'working_dir="/opt/iqa/iqa-mlops"' in source
    assert '"repo_root": "/opt/iqa/iqa-mlops"' in source
    assert '"image_root": "/opt/iqa/iqa-mlops/data/raw/hss-iad"' in source
    assert "pool=GPU_POOL" in source
    assert "max_active_runs=1" in source
    assert "execution_timeout=timedelta(hours=6)" in source
    assert "retries=0" in source


@pytest.mark.unit
def test_lifecycle_dag_declares_comparative_promotion_params() -> None:
    """Lifecycle DAG exposes the fair promotion and registry-relevant params."""
    source = _read_dag_source("iqa_lifecycle.py")

    assert '"promotion_min_delta": 0.0' in source
    assert '"gate_eval_profile": "fast"' in source
    assert '"max_steps": None' in source
    assert '"require_mlflow_registry": False' in source
    assert '"mlflow_tracking_uri": "http://mlflow:5000"' in source
    assert '"MLFLOW_TRACKING_URI": "{{ params.mlflow_tracking_uri }}"' in source
    assert "--promotion-min-delta" in source
    assert "--gate-eval-profile" in source
    assert "--reference-eval-manifest" in source
    assert "--reference-gt-masks-manifest" in source
    assert "--max-steps" in source
    assert "--require-mlflow-registry" in source
    assert "params.require_mlflow_registry in [true, 'True', 'true', '1', 1]" in source


@pytest.mark.unit
def test_lifecycle_dag_keeps_scheduler_free_of_runtime_and_legacy_tasks() -> None:
    """Lifecycle DAG stays lightweight and does not instantiate legacy tasks."""
    source = _read_dag_source("iqa_lifecycle.py")

    assert "PythonOperator(" not in source
    assert "lifecycle_tasks" not in source
    for forbidden in ["import torch", "import pandas", "from iqa.inference", "from iqa.training"]:
        assert forbidden not in source


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
    assert "make_container_task(" in source
    assert "iqa-check-dvc-reproducibility" in source
    # Containerised via the factory on the dedicated dvc-gate image (ADR 0008):
    # booleans pass as templated values, no shell-conditional flags.
    assert "dvc_image()" in source
    assert '"--with-network", "{{ params.with_network }}"' in source
    assert '"--skip-regeneration", "{{ params.skip_regeneration }}"' in source
    assert '"--dvc-target", "{{ params.dvc_target }}"' in source
    assert "{% if params.with_network %}" not in source  # no shell-conditional flags
    assert "BashOperator(" not in source
    assert "bash_command" not in source
    assert "dvc push" not in source


@pytest.mark.docker_contract
def test_iqa_lifecycle_dag_has_single_application_task() -> None:
    """Test that iqa_lifecycle DAG runs the lifecycle as one application task."""
    try:
        import iqa_lifecycle
    except ImportError as e:
        pytest.skip(f"Airflow not installed: {e}")

    dag = iqa_lifecycle.dag
    if dag is None:
        pytest.skip("DAG is None (Airflow not available)")

    assert {task.task_id for task in dag.tasks} == {"run_application_lifecycle"}


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
    assert len(dag.tasks) == 1, f"Expected 1 task, got {len(dag.tasks)}"
    assert dag.get_task("run_application_lifecycle") is not None


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
    for relayed_param in [
        '"gate_eval_profile": "{{ params.gate_eval_profile }}"',
        '"reference_eval_manifest": "{{ params.reference_eval_manifest }}"',
        '"reference_gt_masks_manifest": "{{ params.reference_gt_masks_manifest }}"',
        '"max_steps": "{{ params.max_steps }}"',
        '"require_mlflow_registry": "{{ params.require_mlflow_registry }}"',
        '"mlflow_tracking_uri": "{{ params.mlflow_tracking_uri }}"',
        '"ml_image": "{{ params.ml_image }}"',
    ]:
        assert relayed_param in trigger
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
