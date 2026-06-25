"""IQA drift sensor DAG (ADR 0010, issue 05).

Event-driven half of the drift chain: instead of waiting for the ``@hourly``
poll of ``iqa_lifecycle_trigger``, this DAG watches the Prometheus ``ALERTS``
series and fires ``iqa_lifecycle`` as soon as the calibrated drift alert
(``IqaDriftProxy``, issue 04) goes *firing*.

Design (proposition decision 6, RÉVISÉ) -- the two riskiest components of the
deferrable approach are removed:

- ``PythonSensor mode="reschedule"`` (``poke_interval=15``): event-driven yet
  releases the worker slot between pokes -- **no** ``triggerer`` service, **no**
  custom async ``Trigger`` class.
- The poke is pure stdlib (``urllib``/``json``): it reads the shared Prometheus
  rule via ``GET /api/v1/query`` and the Airflow REST API for the re-trigger
  guard. **No ``iqa`` import in the scheduler** (ADR 0008) -- the threshold lives
  once, in the Prometheus rule, shared by notif/panel/sensor (decision 7).

Two guards keep the chain from queueing a storm of retrains while the replay
keeps ``ALERTS`` *firing* (decisions 13/17):

- **Anti-rejeu**: the poke does not succeed while an ``iqa_lifecycle`` run is
  already ``running``/``queued`` (REST ``dagRuns``); complements
  ``max_active_runs=1``. ``TriggerDagRunOperator(reset_dag_run=False)`` never
  overwrites a live run.
- **Cooldown post-promotion**: after a successful lifecycle run, the poke stays
  quiet for ``cooldown_seconds`` -- bounds the degenerate case where ``ALERTS``
  has not yet fallen back despite the model reload.

``schedule="@continuous"`` (Airflow ≥ 2.6; the project pins ≥ 2.10) keeps a
single sensor run alive and immediately relaunches one after each trigger. On an
older Airflow, fall back to the ``*/2 * * * *`` cron -- the ``reschedule`` mode
does the heavy lifting between relaunches anyway.
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

LIFECYCLE_DAG_ID = "iqa_lifecycle"
ALERT_NAME = "IqaDriftProxy"
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
    """True iff ``ALERTS{alertname=IqaDriftProxy,alertstate=firing}`` has a sample.

    Reads the shared rule via the Prometheus instant-query API -- the threshold
    is never duplicated in the sensor (decision 7).
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


def _lifecycle_dag_runs(states: list[str], *, limit: int = 1, order_by: str | None = None) -> list[dict]:
    """List ``iqa_lifecycle`` dag runs in ``states`` via the Airflow REST API."""
    params: list[tuple[str, str]] = [("state", state) for state in states]
    params.append(("limit", str(limit)))
    if order_by is not None:
        params.append(("order_by", order_by))
    url = (
        f"{_airflow_api_base_url()}/dags/{LIFECYCLE_DAG_ID}/dagRuns"
        f"?{urllib.parse.urlencode(params)}"
    )
    try:
        payload = _http_get_json(url, headers=_airflow_api_headers())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        # API unreachable: be conservative -- treat as "a run may be active" so
        # we never queue a storm we cannot see. Caller decides the policy.
        return [{"__unknown__": True}]
    return payload.get("dag_runs", [])


def _lifecycle_run_in_flight() -> bool:
    """True iff an ``iqa_lifecycle`` run is already running or queued (decision 13)."""
    runs = _lifecycle_dag_runs(["queued", "running"], limit=1)
    return bool(runs)


def _in_cooldown(cooldown_seconds: int) -> bool:
    """True iff the last successful lifecycle run ended < ``cooldown_seconds`` ago.

    Bounds the degenerate case where ``ALERTS`` has not yet fallen back after the
    model reload (decision 17). Disabled when ``cooldown_seconds`` <= 0.
    """
    if cooldown_seconds <= 0:
        return False
    runs = _lifecycle_dag_runs(["success"], limit=1, order_by="-end_date")
    if not runs or runs[0].get("__unknown__"):
        return False
    end_date = runs[0].get("end_date")
    if not end_date:
        return False
    ended = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
    elapsed = (datetime.now(timezone.utc) - ended).total_seconds()
    return elapsed < cooldown_seconds


def _drift_alert_should_trigger(**context) -> bool:
    """Poke callable: succeed only when a retrain is warranted.

    Returns True iff (a) ``IqaDriftProxy`` is firing, (b) no ``iqa_lifecycle``
    run is already in flight (anti-rejeu, decision 13) and (c) we are past the
    post-promotion cooldown (decision 17). The guards run before/after the alert
    read so a firing alert never queues a second run.
    """
    params = (context.get("params") or {})
    cooldown_seconds = int(params.get("cooldown_seconds", 900))

    if not _alert_is_firing():
        return False
    if _lifecycle_run_in_flight():
        return False
    if _in_cooldown(cooldown_seconds):
        return False
    return True


def _define() -> None:
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    from airflow.sensors.python import PythonSensor

    op_wait_for_drift = PythonSensor(
        task_id="wait_for_drift_alert",
        python_callable=_drift_alert_should_trigger,
        mode="reschedule",  # release the worker slot between pokes (decision 6)
        poke_interval=15,
        # Long timeout: under @continuous the run is relaunched after each fire;
        # a timeout simply ends the run and @continuous starts the next one.
        timeout=60 * 60 * 24,
    )

    op_trigger_lifecycle = TriggerDagRunOperator(
        task_id="trigger_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        # Never overwrite a live run -- the anti-rejeu guard already gated us
        # (decision 13).
        reset_dag_run=False,
        conf={
            "scenario_id": DRIFT_SCENARIO_ID,
            "drift_confirmed": "True",
            # Demo-friendly retrain: fast train-on-trigger on GPU so the chain is
            # watchable end-to-end (the reference flow uses progressive-train).
            "mode": "train-on-trigger",
            "max_events": 8,
            "epochs": 1,
            "max_cycles": 1,
        },
    )

    op_wait_for_drift >> op_trigger_lifecycle


dag = build_container_dag(
    dag_id="iqa_drift_sensor",
    define=_define,
    schedule=SENSOR_SCHEDULE,
    # @continuous requires a single live run.
    max_active_runs=1,
    tags=["iqa", "drift", "sensor"],
    params={
        # Post-promotion cooldown (seconds) -- 0 disables it (decision 17).
        # Lowered for the watchable demo so the class2 then class3 retrains can
        # both fire within one session (reference default was 900).
        "cooldown_seconds": 120,
    },
)
