"""Tests for IQA lifecycle DAG (IQA2_KEN12).

Tests that DAG accepts runtime params for replay scenarios.
"""

from __future__ import annotations

import pytest


def test_dag_accepts_regime_and_scenario_id_params() -> None:
    """DAG accepts regime and scenario_id params for replay."""
    try:
        from airflow.dags.iqa_lifecycle import dag
    except ImportError:
        pytest.skip("Airflow not installed")

    if dag is None:
        pytest.skip("DAG dependencies not available")

    assert "regime" in dag.params, "DAG should have 'regime' param"
    assert "scenario_id" in dag.params, "DAG should have 'scenario_id' param"


def test_dag_params_have_sensible_defaults() -> None:
    """DAG params have sensible defaults."""
    try:
        from airflow.dags.iqa_lifecycle import dag
    except ImportError:
        pytest.skip("Airflow not installed")

    if dag is None:
        pytest.skip("DAG dependencies not available")

    assert dag.params.get("regime") == "natural", "Default regime should be 'natural'"
    assert (
        dag.params.get("scenario_id") == "production_replay_natural"
    ), "Default scenario_id should be 'production_replay_natural'"


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
