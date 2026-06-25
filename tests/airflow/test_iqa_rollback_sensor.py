"""Tests for the rollback sensor DAG (Issue 5).

Mirrors the drift sensor tests: source-contract tests (always run) and behaviour
tests for the stdlib poke logic (alert read + anti-rejeu + cooldown), importing
the module and monkeypatching the HTTP helpers -- no Airflow needed.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

DAG_FOLDER = Path(__file__).parents[2] / "airflow" / "dags"
sys.path.insert(0, str(DAG_FOLDER))


def _read_dag_source(name: str = "iqa_rollback_sensor.py") -> str:
    return (DAG_FOLDER / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Source-contract tests
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sensor_dag_uses_reschedule_python_sensor() -> None:
    source = _read_dag_source()
    assert "PythonSensor(" in source
    assert 'mode="reschedule"' in source
    assert "poke_interval=15" in source
    assert "defer(" not in source
    assert "BaseTrigger" not in source


@pytest.mark.unit
def test_sensor_dag_reads_regression_alert_via_prometheus_query_api() -> None:
    source = _read_dag_source()
    assert "/api/v1/query" in source
    assert 'ALERT_NAME = "IqaModelRegression"' in source
    assert 'alertname="{ALERT_NAME}"' in source
    assert 'alertstate="firing"' in source


@pytest.mark.unit
def test_sensor_dag_triggers_rollback_dag() -> None:
    source = _read_dag_source()
    assert "TriggerDagRunOperator(" in source
    assert 'trigger_dag_id=ROLLBACK_DAG_ID' in source or 'trigger_dag_id="iqa_rollback"' in source
    assert "reset_dag_run=False" in source
    assert "op_wait_for_regression >> op_trigger_rollback" in source


@pytest.mark.unit
def test_sensor_dag_stays_metier_free_and_stdlib_only() -> None:
    source = _read_dag_source()
    assert "import urllib" in source
    assert "import json" in source
    for forbidden in ["import torch", "import pandas", "from iqa.inference", "from iqa.training", "from iqa.promotion"]:
        assert forbidden not in source
    assert "BashOperator(" not in source


@pytest.mark.unit
def test_sensor_dag_schedule_compatible_with_airflow_version() -> None:
    source = _read_dag_source()
    assert 'SENSOR_SCHEDULE = "@continuous"' in source
    assert "max_active_runs=1" in source
    assert "*/2 * * * *" in source  # documented fallback


# --------------------------------------------------------------------------- #
# Behaviour tests for the poke logic
# --------------------------------------------------------------------------- #


@pytest.fixture
def sensor_module(monkeypatch):
    import iqa_rollback_sensor as mod

    monkeypatch.setenv("IQA_PROMETHEUS_URL", "http://prom:9090")
    monkeypatch.setenv("IQA_AIRFLOW_API_URL", "http://af:8080/api/v1")
    return mod


def test_alert_is_firing_true_when_result_present(sensor_module, monkeypatch):
    monkeypatch.setattr(
        sensor_module, "_http_get_json",
        lambda *a, **k: {"status": "success", "data": {"result": [{"metric": {}}]}},
    )
    assert sensor_module._alert_is_firing() is True


def test_alert_is_firing_false_when_no_result(sensor_module, monkeypatch):
    monkeypatch.setattr(
        sensor_module, "_http_get_json",
        lambda *a, **k: {"status": "success", "data": {"result": []}},
    )
    assert sensor_module._alert_is_firing() is False


def test_rollback_run_in_flight_true_when_run_present(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_http_get_json", lambda *a, **k: {"dag_runs": [{"state": "running"}]})
    assert sensor_module._rollback_run_in_flight() is True


def test_rollback_run_in_flight_conservative_on_api_error(sensor_module, monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(sensor_module, "_http_get_json", boom)
    assert sensor_module._rollback_run_in_flight() is True


def test_in_cooldown_true_when_recent_success(sensor_module, monkeypatch):
    recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    monkeypatch.setattr(sensor_module, "_http_get_json", lambda *a, **k: {"dag_runs": [{"end_date": recent}]})
    assert sensor_module._in_cooldown(900) is True


def test_in_cooldown_disabled_when_zero(sensor_module, monkeypatch):
    called = False

    def spy(*a, **k):
        nonlocal called
        called = True
        return {"dag_runs": []}

    monkeypatch.setattr(sensor_module, "_http_get_json", spy)
    assert sensor_module._in_cooldown(0) is False
    assert called is False


def test_should_trigger_only_when_firing_and_no_guard(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: True)
    monkeypatch.setattr(sensor_module, "_rollback_run_in_flight", lambda: False)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: False)
    assert sensor_module._regression_alert_should_trigger(params={}) is True


def test_should_not_trigger_when_alert_quiet(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: False)
    monkeypatch.setattr(sensor_module, "_rollback_run_in_flight", lambda: False)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: False)
    assert sensor_module._regression_alert_should_trigger(params={}) is False


def test_should_not_trigger_when_run_in_flight(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: True)
    monkeypatch.setattr(sensor_module, "_rollback_run_in_flight", lambda: True)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: False)
    assert sensor_module._regression_alert_should_trigger(params={}) is False


def test_should_not_trigger_during_cooldown(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: True)
    monkeypatch.setattr(sensor_module, "_rollback_run_in_flight", lambda: False)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: True)
    assert sensor_module._regression_alert_should_trigger(params={"cooldown_seconds": 900}) is False
