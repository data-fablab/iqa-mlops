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
   ``iqa_lifecycle`` and forwards the raw signal plus the application lifecycle
   parameters as ``conf``.

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
            "image_root": "{{ params.image_root }}",
            "mode": "{{ params.mode }}",
            "max_events": "{{ params.max_events }}",
            "lifecycle_interval": "{{ params.lifecycle_interval }}",
            "max_cycles": "{{ params.max_cycles }}",
            "epochs": "{{ params.epochs }}",
            "target_stage": "{{ params.target_stage }}",
            "promotion_min_delta": "{{ params.promotion_min_delta }}",
            "anchor_good_manifest": "{{ params.anchor_good_manifest }}",
            "anchor_good_max_per_class": "{{ params.anchor_good_max_per_class }}",
            "hard_good_max_per_class": "{{ params.hard_good_max_per_class }}",
            "reference_eval_manifest": "{{ params.reference_eval_manifest }}",
            "reference_gt_masks_manifest": "{{ params.reference_gt_masks_manifest }}",
            "progressive_min_defects_for_decision": "{{ params.progressive_min_defects_for_decision }}",
            "max_good_alert_rate": "{{ params.max_good_alert_rate }}",
            "max_good_red_rate": "{{ params.max_good_red_rate }}",
            "candidate_init_policy": "{{ params.candidate_init_policy }}",
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
        "image_root": "/opt/iqa/iqa-mlops/data/raw/hss-iad",
        "mode": "progressive-train",
        "max_events": 260,
        "lifecycle_interval": 50,
        "max_cycles": 3,
        "epochs": 10,
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "anchor_good_manifest": "data/model_datasets/feature_ae_good_v002.csv",
        "anchor_good_max_per_class": 256,
        "hard_good_max_per_class": 64,
        "reference_eval_manifest": "data/validation/validation_set_v001.csv",
        "reference_gt_masks_manifest": "data/validation/validation_gt_masks_v001.csv",
        "progressive_min_defects_for_decision": 5,
        "max_good_alert_rate": 0.10,
        "max_good_red_rate": 0.02,
        "candidate_init_policy": "stable_base",
        "image": data_image(),
    },
)
