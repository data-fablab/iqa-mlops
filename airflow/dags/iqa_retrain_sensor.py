"""IQA retrain policy sensor DAG (Issues 15-18, ADR 0010 amendment).

Periodic pull-based sensor that assembles a ``RetrainPolicySignal`` from
three sources (accumulation counter, prod metrics via MLflow, PatchCore
drift via Prometheus) and calls the pure ``evaluate_retrain_policy``
function. On trigger, launches ``iqa_lifecycle`` with the resolved conf.

Idempotent: does not re-trigger if a lifecycle run is already active for
the same reason. The webhook-catcher remains passive (observability only).
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
SENSOR_SCHEDULE = "*/5 * * * *"

STATE_FILE = "/var/lib/iqa/retrain_policy_state.json"


def _prometheus_base_url() -> str:
    return os.environ.get("IQA_PROMETHEUS_URL", "http://prometheus:9090").rstrip("/")


def _airflow_api_base_url() -> str:
    return os.environ.get(
        "IQA_AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1"
    ).rstrip("/")


def _http_get_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 10) -> dict:
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _airflow_api_headers() -> dict[str, str]:
    user = os.environ.get("IQA_AIRFLOW_API_USER", "airflow")
    password = os.environ.get("IQA_AIRFLOW_API_PASSWORD", "airflow")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _lifecycle_run_active() -> bool:
    params = urllib.parse.urlencode([("state", "running"), ("state", "queued"), ("limit", "1")])
    url = f"{_airflow_api_base_url()}/dags/{LIFECYCLE_DAG_ID}/dagRuns?{params}"
    try:
        payload = _http_get_json(url, headers=_airflow_api_headers())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return True
    return bool(payload.get("dag_runs", []))


def _fetch_accumulation_count() -> int:
    promql = 'iqa_conforming_validated_total'
    url = f"{_prometheus_base_url()}/api/v1/query?{urllib.parse.urlencode({'query': promql})}"
    try:
        payload = _http_get_json(url)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return 0
    results = payload.get("data", {}).get("result", [])
    if not results:
        return 0
    try:
        return int(float(results[0]["value"][1]))
    except (KeyError, IndexError, ValueError, TypeError):
        return 0


def _fetch_drift_signal() -> tuple[bool, str | None, float]:
    promql = 'iqa_domain_drift_total{regime="out_of_domain"}'
    url = f"{_prometheus_base_url()}/api/v1/query?{urllib.parse.urlencode({'query': promql})}"
    try:
        payload = _http_get_json(url)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False, None, 0.0
    results = payload.get("data", {}).get("result", [])
    if not results:
        return False, None, 0.0

    best_class = None
    best_count = 0.0
    total_count = 0.0
    for series in results:
        source_class = series.get("metric", {}).get("source_class", "")
        try:
            count = float(series["value"][1])
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        total_count += count
        if count > best_count:
            best_count = count
            best_class = source_class

    total_promql = 'iqa_domain_drift_total'
    total_url = f"{_prometheus_base_url()}/api/v1/query?{urllib.parse.urlencode({'query': total_promql})}"
    total_all = 0.0
    try:
        total_payload = _http_get_json(total_url)
        for series in total_payload.get("data", {}).get("result", []):
            try:
                total_all += float(series["value"][1])
            except (KeyError, IndexError, ValueError, TypeError):
                continue
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass

    ood_ratio = best_count / total_all if total_all > 0 else 0.0

    for alert_name in ("IqaDomainDriftPatchCore", "IqaDriftProxy"):
        alert_promql = f'ALERTS{{alertname="{alert_name}",alertstate="firing"}}'
        alert_url = f"{_prometheus_base_url()}/api/v1/query?{urllib.parse.urlencode({'query': alert_promql})}"
        try:
            alert_payload = _http_get_json(alert_url)
            if alert_payload.get("data", {}).get("result"):
                return True, best_class, ood_ratio
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            continue

    return False, best_class, ood_ratio


def _fetch_prod_metrics() -> dict[str, float]:
    try:
        from iqa.monitoring.model_metrics import fetch_latest_quality_metrics
        return fetch_latest_quality_metrics("prod")
    except Exception:
        return {}


def _load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass


def _evaluate_and_trigger(**context) -> bool:
    from iqa.monitoring.retrain_policy import (
        RetrainPolicySignal,
        evaluate_retrain_policy,
    )

    active = _lifecycle_run_active()
    count = _fetch_accumulation_count()
    drift_confirmed, triggering_class, ood_ratio = _fetch_drift_signal()
    prod_metrics = _fetch_prod_metrics()

    state = _load_state()
    last_inputs = state.get("last_trigger_inputs")
    gate_failures = state.get("gate_failure_count", 0)
    last_attempt_ts = state.get("last_attempt_ts")
    seconds_since = None
    if last_attempt_ts:
        try:
            seconds_since = (datetime.now(timezone.utc) - datetime.fromisoformat(last_attempt_ts)).total_seconds()
        except (ValueError, TypeError):
            pass

    signal = RetrainPolicySignal(
        conforming_validated_count=count,
        drift_confirmed=drift_confirmed,
        drift_triggering_class=triggering_class,
        drift_ood_ratio=ood_ratio,
        prod_metrics=prod_metrics,
        last_trigger_inputs=last_inputs,
        gate_failure_count=gate_failures,
        seconds_since_last_attempt=seconds_since,
        active_lifecycle_run=active,
    )

    decision = evaluate_retrain_policy(signal)

    ti = context.get("ti")
    if ti is not None:
        ti.xcom_push(key="decision", value=decision.to_dict())

    if decision.trigger:
        state["last_trigger_inputs"] = {
            "conforming_validated_count": count,
            "drift_triggering_class": triggering_class,
            "prod_metrics": prod_metrics,
        }
        state["last_attempt_ts"] = datetime.now(timezone.utc).isoformat()
        _save_state(state)

        if ti is not None:
            ti.xcom_push(key="triggering_class", value=triggering_class or "Casting_class1")
            ti.xcom_push(key="retrain_scope", value=decision.retrain_scope)
            ti.xcom_push(key="trigger_reason", value=decision.primary_reason)
    return decision.trigger


def _define() -> None:
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    op_evaluate = ShortCircuitOperator(
        task_id="evaluate_retrain_policy",
        python_callable=_evaluate_and_trigger,
    )

    op_trigger = TriggerDagRunOperator(
        task_id="trigger_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        reset_dag_run=False,
        conf={
            "scenario_id": "retrain_policy",
            "trigger_reason": "{{ ti.xcom_pull(task_ids='evaluate_retrain_policy', key='trigger_reason') or 'accumulation' }}",
            "triggering_class": "{{ ti.xcom_pull(task_ids='evaluate_retrain_policy', key='triggering_class') or 'Casting_class1' }}",
            "retrain_scope": "{{ ti.xcom_pull(task_ids='evaluate_retrain_policy', key='retrain_scope') or 'bootstrap' }}",
            "drift_confirmed": "{{ ti.xcom_pull(task_ids='evaluate_retrain_policy', key='trigger_reason') == 'drift' }}",
            "mode": "train-on-trigger",
            "max_events": 8,
            "epochs": 1,
            "max_cycles": 1,
        },
    )

    op_evaluate >> op_trigger


dag = build_container_dag(
    dag_id="iqa_retrain_sensor",
    define=_define,
    schedule=SENSOR_SCHEDULE,
    max_active_runs=1,
    tags=["iqa", "retrain", "sensor", "policy"],
    params={
        "cooldown_seconds": 900,
    },
)
