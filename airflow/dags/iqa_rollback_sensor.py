"""IQA rollback sensor DAG (Issue 5).

Event-driven trigger of the rollback path: it watches the Prometheus ``ALERTS``
series and fires ``iqa_rollback`` as soon as the metric-regression alert
(``IqaModelRegression``, Issue 5) goes *firing*. Mirrors ``iqa_drift_sensor``
(ADR 0010, issue 05) so notif/panel/sensor share the single threshold that lives
in the Prometheus rule (decision 7) -- the sensor never duplicates it.

Same deliberately-boring mechanics as the drift sensor:

- ``PythonSensor mode="reschedule"`` (``poke_interval=15``): event-driven yet
  releases the worker slot between pokes -- **no** triggerer, **no** custom async
  Trigger.
- The poke is pure stdlib (``urllib``/``json``): it reads the shared rule via
  ``GET /api/v1/query`` and the Airflow REST API for the re-trigger guard. **No
  ``iqa`` import in the scheduler** (ADR 0008).

Two guards keep a firing alert from queueing a storm of rollbacks while ``ALERTS``
stays firing:

- **Anti-rejeu**: the poke does not succeed while an ``iqa_rollback`` run is
  already ``running``/``queued``; complements ``max_active_runs=1``.
  ``TriggerDagRunOperator(reset_dag_run=False)`` never overwrites a live run.
- **Cooldown post-rollback**: after a successful rollback the poke stays quiet
  for ``cooldown_seconds`` -- bounds the case where ``ALERTS`` has not yet fallen
  back despite the restored model being reloaded.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from iqa.dags import build_container_dag

ROLLBACK_DAG_ID = "iqa_rollback"
ALERT_NAME = "IqaModelRegression"
DRIFT_SCENARIO_ID = "drift_domain_extension"

# @continuous needs Airflow >= 2.6 (project pins >= 2.10). On older Airflow,
# replace with "*/2 * * * *" -- reschedule does the work between relaunches.
SENSOR_SCHEDULE = "@continuous"


def _prometheus_base_url() -> str:
    """Prometheus base URL (override per deploy)."""
    return os.environ.get("IQA_PROMETHEUS_URL", "http://prometheus:9090").rstrip("/")


def _airflow_api_base_url() -> str:
    """Airflow stable REST API base URL (override per deploy)."""
    return os.environ.get(
        "IQA_AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1"
    ).rstrip("/")


def _http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 10) -> dict:
    """GET ``url`` and parse the JSON body. Stdlib only (ADR 0008)."""
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - internal hosts
        return json.loads(response.read().decode("utf-8"))


def _alert_is_firing() -> bool:
    """True iff ``ALERTS{alertname=IqaModelRegression,alertstate=firing}`` has a sample.

    Reads the shared rule via the Prometheus instant-query API -- the threshold is
    never duplicated in the sensor (decision 7).
    """
    promql = f'ALERTS{{alertname="{ALERT_NAME}",alertstate="firing"}}'
    url = f"{_prometheus_base_url()}/api/v1/query?{urllib.parse.urlencode({'query': promql})}"
    try:
        payload = _http_get_json(url)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        # Prometheus unreachable / malformed: stay quiet, retry next poke.
        return False
    if payload.get("status") != "success":
        return False
    return bool(payload.get("data", {}).get("result"))


def _airflow_api_headers() -> dict[str, str]:
    """Basic-auth header for the Airflow REST API (creds from env)."""
    user = os.environ.get("IQA_AIRFLOW_API_USER", "airflow")
    password = os.environ.get("IQA_AIRFLOW_API_PASSWORD", "airflow")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _rollback_dag_runs(states: list[str], *, limit: int = 1, order_by: str | None = None) -> list[dict]:
    """List ``iqa_rollback`` dag runs in ``states`` via the Airflow REST API."""
    params: list[tuple[str, str]] = [("state", state) for state in states]
    params.append(("limit", str(limit)))
    if order_by is not None:
        params.append(("order_by", order_by))
    url = (
        f"{_airflow_api_base_url()}/dags/{ROLLBACK_DAG_ID}/dagRuns"
        f"?{urllib.parse.urlencode(params)}"
    )
    try:
        payload = _http_get_json(url, headers=_airflow_api_headers())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        # API unreachable: be conservative -- treat as "a run may be active" so we
        # never queue a storm we cannot see. Caller decides the policy.
        return [{"__unknown__": True}]
    return payload.get("dag_runs", [])


def _rollback_run_in_flight() -> bool:
    """True iff an ``iqa_rollback`` run is already running or queued (anti-rejeu)."""
    runs = _rollback_dag_runs(["queued", "running"], limit=1)
    return bool(runs)


def _in_cooldown(cooldown_seconds: int) -> bool:
    """True iff the last successful rollback ended < ``cooldown_seconds`` ago.

    Bounds the case where ``ALERTS`` has not yet fallen back after the restored
    model is reloaded. Disabled when ``cooldown_seconds`` <= 0.
    """
    if cooldown_seconds <= 0:
        return False
    runs = _rollback_dag_runs(["success"], limit=1, order_by="-end_date")
    if not runs or runs[0].get("__unknown__"):
        return False
    end_date = runs[0].get("end_date")
    if not end_date:
        return False
    ended = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    elapsed = (datetime.now(timezone.utc) - ended).total_seconds()
    return elapsed < cooldown_seconds


def _regression_alert_should_trigger(**context) -> bool:
    """Poke callable: succeed only when a rollback is warranted.

    Returns True iff (a) ``IqaModelRegression`` is firing, (b) no ``iqa_rollback``
    run is already in flight (anti-rejeu) and (c) we are past the post-rollback
    cooldown. The guards run before/after the alert read so a firing alert never
    queues a second run.
    """
    params = context.get("params") or {}
    cooldown_seconds = int(params.get("cooldown_seconds", 900))

    if not _alert_is_firing():
        return False
    if _rollback_run_in_flight():
        return False
    if _in_cooldown(cooldown_seconds):
        return False
    return True


def _define() -> None:
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    from airflow.sensors.python import PythonSensor

    op_wait_for_regression = PythonSensor(
        task_id="wait_for_regression_alert",
        python_callable=_regression_alert_should_trigger,
        mode="reschedule",  # release the worker slot between pokes
        poke_interval=15,
        # Long timeout: under @continuous the run is relaunched after each fire;
        # a timeout simply ends the run and @continuous starts the next one.
        timeout=60 * 60 * 24,
    )

    op_trigger_rollback = TriggerDagRunOperator(
        task_id="trigger_rollback",
        trigger_dag_id=ROLLBACK_DAG_ID,
        # Never overwrite a live run -- the anti-rejeu guard already gated us.
        reset_dag_run=False,
        conf={
            "scenario_id": DRIFT_SCENARIO_ID,
            # Empty -> iqa-run-rollback rolls back from the current prod version.
            "faulty_version": "",
        },
    )

    op_wait_for_regression >> op_trigger_rollback


dag = build_container_dag(
    dag_id="iqa_rollback_sensor",
    define=_define,
    schedule=SENSOR_SCHEDULE,
    # @continuous requires a single live run.
    max_active_runs=1,
    tags=["iqa", "rollback", "sensor"],
    params={
        # Post-rollback cooldown (seconds) -- 0 disables it.
        "cooldown_seconds": 120,
    },
)
