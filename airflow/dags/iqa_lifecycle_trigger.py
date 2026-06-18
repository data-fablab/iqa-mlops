"""IQA event-driven lifecycle trigger DAG (ADR 0002, issue 16).

This is the capstone of the orchestration chain: it closes the loop so the
``iqa_lifecycle`` pipeline starts on a *data event* (drift confirmed, full batch,
or enough oracle-validated conformes) without any manual trigger.

Three steps, every metier decision in a container, the trigger itself native
Airflow glue (no ``iqa`` import in the scheduler, ADR 0008):

1. ``evaluate_decision`` -- runs the ``data`` image with ``iqa-run-lifecycle-decision``
   via the operator factory. The data-event rule (``evaluate_lifecycle_signal``)
   is evaluated **inside the container**; its JSON decision is the task XCom and
   carries ``trigger_lifecycle``.
2. ``gate_on_decision`` -- a ``ShortCircuitOperator``: pure orchestration glue
   (``json`` only, never imports ``iqa``). It reads the container decision and
   short-circuits unless ``trigger_lifecycle`` is true -- so nothing fires on the
   nominal "keep waiting" path.
3. ``trigger_lifecycle`` -- a ``TriggerDagRunOperator`` that launches
   ``iqa_lifecycle`` and forwards the raw signal params as ``conf``. The
   candidate dataset version is re-derived by the target DAG's own
   ``lifecycle_decision`` task, so the trigger only relays the signal.

The thresholds are the existing data-event rule (configurable via params /
``min_natural_conforming``); the real observation of the store state
(PostgreSQL events / monitoring) is the data plane, tracked by the runtime
sisters (18 / 23). This slice wires the automatic trigger, not the polling I/O.
"""

from __future__ import annotations

import json

from iqa.dags import build_container_dag, data_image, make_container_task


DECISION_TASK_ID = "evaluate_decision"
LIFECYCLE_DAG_ID = "iqa_lifecycle"


def _should_trigger(ti=None, **_context) -> bool:
    """Short-circuit glue: trigger the lifecycle iff the container said so.

    Reads the ``evaluate_decision`` container stdout (its JSON decision pushed to
    XCom) and returns ``trigger_lifecycle``. No ``iqa`` import: the rule already
    ran in the container; this only parses the boolean (ADR 0008).
    """
    payload = ti.xcom_pull(task_ids=DECISION_TASK_ID)
    if payload is None:
        return False
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    return bool(payload.get("trigger_lifecycle", False))


def _define() -> None:
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    op_evaluate_decision = make_container_task(
        task_id=DECISION_TASK_ID,
        image="{{ params.image }}",
        command=[
            "iqa-run-lifecycle-decision",
            "--scenario-id", "{{ params.scenario_id }}",
            "--conforming-validated-count", "{{ params.conforming_validated_count }}",
            "--drift-confirmed", "{{ params.drift_confirmed }}",
            "--roi-fail-rate", "{{ params.roi_fail_rate }}",
        ],
    )

    op_gate_on_decision = ShortCircuitOperator(
        task_id="gate_on_decision",
        python_callable=_should_trigger,
    )

    op_trigger_lifecycle = TriggerDagRunOperator(
        task_id="trigger_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        # Relay the raw signal; iqa_lifecycle re-derives the candidate
        # dataset version in its own lifecycle_decision task.
        conf={
            "scenario_id": "{{ params.scenario_id }}",
            "conforming_validated_count": "{{ params.conforming_validated_count }}",
            "drift_confirmed": "{{ params.drift_confirmed }}",
            "roi_fail_rate": "{{ params.roi_fail_rate }}",
            "target_stage": "{{ params.target_stage }}",
        },
    )

    op_evaluate_decision >> op_gate_on_decision >> op_trigger_lifecycle


dag = build_container_dag(
    dag_id="iqa_lifecycle_trigger",
    define=_define,
    schedule="@hourly",
    tags=["iqa", "lifecycle", "trigger"],
    params={
        "scenario_id": "production_replay_natural",
        "conforming_validated_count": 0,
        "drift_confirmed": False,
        "roi_fail_rate": 0.0,
        "target_stage": "test",
        "image": data_image(),
    },
)
