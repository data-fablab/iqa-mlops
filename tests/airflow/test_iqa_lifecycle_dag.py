"""Tests for IQA lifecycle DAG (IQA2_KEN12).

Tests that DAG accepts runtime params for replay scenarios.
"""

from __future__ import annotations

import pytest

# Every test here resolves the live DAG object, which only exists inside the
# Airflow runtime (Docker image). Skipped locally; selectable via `-m docker_contract`.
pytestmark = pytest.mark.docker_contract


def test_dag_accepts_regime_and_scenario_id_params() -> None:
    """DAG accepts regime and scenario_id params for replay."""
    try:
        from airflow.dags.iqa_lifecycle import dag
    except ImportError:
        pytest.skip("Airflow not installed")

    if dag is None:
        pytest.skip("DAG dependencies not available")

    assert "scenario_id" in dag.params, "DAG should have 'scenario_id' param"
    assert "image_root" in dag.params
    assert "mode" in dag.params
    assert "max_events" in dag.params
    assert "lifecycle_interval" in dag.params
    assert "max_cycles" in dag.params
    assert "epochs" in dag.params
    assert "promotion_min_delta" in dag.params
    assert "target_stage" in dag.params


def test_dag_params_have_sensible_defaults() -> None:
    """DAG params have sensible defaults."""
    try:
        from airflow.dags.iqa_lifecycle import dag
    except ImportError:
        pytest.skip("Airflow not installed")

    if dag is None:
        pytest.skip("DAG dependencies not available")

    assert (
        dag.params.get("scenario_id") == "production_replay_natural"
    ), "Default scenario_id should be 'production_replay_natural'"
    assert dag.params.get("mode") == "progressive-train"
    assert dag.params.get("max_events") == 260
    assert dag.params.get("lifecycle_interval") == 50
    assert dag.params.get("max_cycles") == 3
    assert dag.params.get("epochs") == 10
    assert dag.params.get("target_stage") == "test"
    assert dag.params.get("promotion_min_delta") == 0.0
    assert dag.params.get("require_mlflow_registry") is False


def test_dag_can_run_with_drift_scenario() -> None:
    """DAG can be triggered with drift scenario params."""
    try:
        from airflow.dags.iqa_lifecycle import dag
    except ImportError:
        pytest.skip("Airflow not installed")

    if dag is None:
        pytest.skip("DAG dependencies not available")

    # In Airflow, params flow through context to tasks
    # This test verifies the DAG structure supports it
    assert dag.params.get("regime") is not None
    assert dag.params.get("scenario_id") is not None
    # Tasks can later read these from context in actual dag runs
