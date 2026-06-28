"""Tests for the drift sensor DAG (issue 05).

Two layers: source-contract tests (mirror the other DAG contract tests, always
run) and behaviour tests for the stdlib poke logic (alert read + anti-rejeu +
cooldown), which import the module and monkeypatch the HTTP helpers -- no Airflow
needed.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

DAG_FOLDER = Path(__file__).parents[2] / "airflow" / "dags"
sys.path.insert(0, str(DAG_FOLDER))


def _read_dag_source(name: str = "iqa_drift_sensor.py") -> str:
    return (DAG_FOLDER / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Source-contract tests
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_sensor_dag_uses_reschedule_python_sensor() -> None:
    """PythonSensor in reschedule mode, 15s poke -- no triggerer (decision 6)."""
    source = _read_dag_source()

    assert "PythonSensor(" in source
    assert 'mode="reschedule"' in source
    assert "poke_interval=15" in source
    # No deferrable/triggerer machinery in the code (decision 6).
    assert "defer(" not in source
    assert "BaseTrigger" not in source
    assert "deferrable=True" not in source


@pytest.mark.unit
def test_sensor_dag_reads_alerts_via_prometheus_query_api() -> None:
    """The poke reads the shared ALERTS series via the Prometheus query API."""
    source = _read_dag_source()

    assert "/api/v1/query" in source
    assert 'ALERT_NAME = "IqaDriftProxy"' in source
    assert 'PATCHCORE_ALERT_NAME = "IqaDomainDriftPatchCore"' in source
    assert 'alertname="{alert}"' in source
    assert 'alertstate="firing"' in source


@pytest.mark.unit
def test_sensor_dag_triggers_lifecycle_with_drift_conf() -> None:
    """A firing alert triggers iqa_lifecycle with the drift scenario conf."""
    source = _read_dag_source()

    assert "TriggerDagRunOperator(" in source
    assert 'trigger_dag_id=LIFECYCLE_DAG_ID' in source or 'trigger_dag_id="iqa_lifecycle"' in source
    assert '"scenario_id": DRIFT_SCENARIO_ID' in source or '"drift_domain_extension"' in source
    assert '"drift_confirmed": "True"' in source
    assert "reset_dag_run=False" in source
    assert "op_wait_for_drift >> op_trigger_lifecycle" in source


@pytest.mark.unit
def test_sensor_dag_stays_metier_free_and_stdlib_only() -> None:
    """No iqa runtime import, no shell -- urllib/json only (ADR 0008)."""
    source = _read_dag_source()

    assert "import urllib" in source
    assert "import json" in source
    for forbidden in ["import torch", "import pandas", "from iqa.inference", "from iqa.training"]:
        assert forbidden not in source
    assert "BashOperator(" not in source
    assert "bash_command" not in source


@pytest.mark.unit
def test_sensor_dag_schedule_compatible_with_airflow_version() -> None:
    """Schedule is @continuous (Airflow >= 2.6) with the cron fallback documented."""
    source = _read_dag_source()

    assert 'SENSOR_SCHEDULE = "@continuous"' in source
    assert "max_active_runs=1" in source
    assert "*/2 * * * *" in source  # documented fallback


# --------------------------------------------------------------------------- #
# Behaviour tests for the poke logic
# --------------------------------------------------------------------------- #


@pytest.fixture
def sensor_module(monkeypatch):
    import iqa_drift_sensor as mod

    # Deterministic endpoints; the HTTP layer is monkeypatched per test.
    monkeypatch.setenv("IQA_PROMETHEUS_URL", "http://prom:9090")
    monkeypatch.setenv("IQA_AIRFLOW_API_URL", "http://af:8080/api/v1")
    return mod


def test_alert_is_firing_true_when_result_present(sensor_module, monkeypatch):
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: {"status": "success", "data": {"result": [{"metric": {}}]}},
    )
    assert sensor_module._alert_is_firing() is True


def test_alert_is_firing_false_when_no_result(sensor_module, monkeypatch):
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: {"status": "success", "data": {"result": []}},
    )
    assert sensor_module._alert_is_firing() is False


def test_alert_is_firing_false_on_http_error(sensor_module, monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(sensor_module, "_http_get_json", boom)
    assert sensor_module._alert_is_firing() is False


def test_lifecycle_run_in_flight_true_when_run_present(sensor_module, monkeypatch):
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: {"dag_runs": [{"state": "running"}]},
    )
    assert sensor_module._lifecycle_run_in_flight() is True


def test_lifecycle_run_in_flight_false_when_none(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_http_get_json", lambda *a, **k: {"dag_runs": []})
    assert sensor_module._lifecycle_run_in_flight() is False


def test_lifecycle_run_in_flight_conservative_on_api_error(sensor_module, monkeypatch):
    """API unreachable -> treat as in-flight so we never queue a storm blind."""
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(sensor_module, "_http_get_json", boom)
    assert sensor_module._lifecycle_run_in_flight() is True


def test_in_cooldown_true_when_recent_success(sensor_module, monkeypatch):
    recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: {"dag_runs": [{"end_date": recent}]},
    )
    assert sensor_module._in_cooldown(900) is True


def test_in_cooldown_false_when_old_success(sensor_module, monkeypatch):
    old = (datetime.now(timezone.utc) - timedelta(seconds=1800)).isoformat()
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: {"dag_runs": [{"end_date": old}]},
    )
    assert sensor_module._in_cooldown(900) is False


def test_in_cooldown_disabled_when_zero(sensor_module, monkeypatch):
    called = False

    def spy(*a, **k):
        nonlocal called
        called = True
        return {"dag_runs": []}

    monkeypatch.setattr(sensor_module, "_http_get_json", spy)
    assert sensor_module._in_cooldown(0) is False
    assert called is False  # short-circuits before any HTTP call


def test_should_trigger_only_when_firing_and_no_guard(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: True)
    monkeypatch.setattr(sensor_module, "_lifecycle_run_in_flight", lambda: False)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: False)
    assert sensor_module._drift_alert_should_trigger(params={}) is True


def test_should_not_trigger_when_alert_quiet(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: False)
    monkeypatch.setattr(sensor_module, "_lifecycle_run_in_flight", lambda: False)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: False)
    assert sensor_module._drift_alert_should_trigger(params={}) is False


def test_should_not_trigger_when_run_in_flight(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: True)
    monkeypatch.setattr(sensor_module, "_lifecycle_run_in_flight", lambda: True)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: False)
    assert sensor_module._drift_alert_should_trigger(params={}) is False


def test_should_not_trigger_during_cooldown(sensor_module, monkeypatch):
    monkeypatch.setattr(sensor_module, "_alert_is_firing", lambda: True)
    monkeypatch.setattr(sensor_module, "_lifecycle_run_in_flight", lambda: False)
    monkeypatch.setattr(sensor_module, "_in_cooldown", lambda s: True)
    assert sensor_module._drift_alert_should_trigger(params={"cooldown_seconds": 900}) is False


# --------------------------------------------------------------------------- #
# Triggering-class detection (issue: class3 mis-attributed to the default)
# --------------------------------------------------------------------------- #


def _drift_series(samples: dict[str, float]) -> dict:
    """Build a Prometheus instant-query payload of per-source_class rate samples."""
    return {
        "status": "success",
        "data": {
            "result": [
                {"metric": {"source_class": cls}, "value": [0, str(rate)]}
                for cls, rate in samples.items()
            ]
        },
    }


def test_detect_triggering_class_picks_highest_ood_rate(sensor_module, monkeypatch):
    """The class with the highest live out-of-domain rate wins (not the default)."""
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: _drift_series({"Casting_class2": 0.1, "Casting_class3": 0.9}),
    )
    assert sensor_module._detect_triggering_class() == "Casting_class3"


def test_detect_triggering_class_queries_rate_not_raw_counter(sensor_module):
    """The PromQL uses a rate window so a covered class's lifetime total cannot win."""
    source = _read_dag_source()
    assert "rate(iqa_domain_drift_total" in source
    assert 'source_class!=""' in source


def test_detect_triggering_class_falls_back_when_no_class_series(sensor_module, monkeypatch):
    monkeypatch.setattr(
        sensor_module,
        "_http_get_json",
        lambda *a, **k: {"status": "success", "data": {"result": []}},
    )
    assert sensor_module._detect_triggering_class() == sensor_module.DEFAULT_TRIGGERING_CLASS


def test_detect_triggering_class_falls_back_on_http_error(sensor_module, monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(sensor_module, "_http_get_json", boom)
    assert sensor_module._detect_triggering_class() == sensor_module.DEFAULT_TRIGGERING_CLASS
