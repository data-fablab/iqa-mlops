"""Poll durable lifecycle signals and trigger the Feature-AE lifecycle.

Every hour, two independent branches collect PostgreSQL-backed signals:

- natural replay: new conforming and train-eligible ``oracle_gt`` feedback;
- drift replay: the latest unconsumed versioned drift observation.

The data-image container evaluates and journals each decision. Native Airflow
glue gates each branch and triggers ``iqa_lifecycle``. ``max_active_runs=1``
prevents overlapping polling runs.
"""

from __future__ import annotations

import json

from iqa.dags import build_container_dag, data_image, make_container_task


NATURAL_DECISION_TASK_ID = "evaluate_decision"
DRIFT_DECISION_TASK_ID = "evaluate_drift_decision"
LIFECYCLE_DAG_ID = "iqa_lifecycle"


def _decision_payload(task_id: str, ti=None) -> dict:
    payload = ti.xcom_pull(task_ids=task_id)
    if payload is None:
        return {}
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _should_trigger(decision_task_id: str, ti=None, **_context) -> bool:
    payload = _decision_payload(decision_task_id, ti)
    return bool(payload.get("trigger_lifecycle", False))


def _collector_task(task_id: str, scenario_param: str):
    return make_container_task(
        task_id=task_id,
        image="{{ params.image }}",
        command=[
            "iqa-collect-lifecycle-signal",
            "--scenario-id",
            "{{ params." + scenario_param + " }}",
            "--roi-window-size",
            "{{ params.roi_window_size }}",
            "--min-natural-conforming",
            "{{ params.min_natural_conforming }}",
        ],
    )


def _lifecycle_conf(
    *,
    decision_task_id: str,
    scenario_param: str,
    candidate_param: str,
    anchor_manifest_param: str,
) -> dict:
    return {
        "scenario_id": "{{ params." + scenario_param + " }}",
        "candidate_dataset_version": "{{ params." + candidate_param + " }}",
        "lifecycle_decision_json": (
            "{{ ti.xcom_pull(task_ids='" + decision_task_id + "') }}"
        ),
        "mode": "{{ params.mode }}",
        "max_events": "{{ params.max_events }}",
        "lifecycle_interval": "{{ params.lifecycle_interval }}",
        "max_cycles": "{{ params.max_cycles }}",
        "epochs": "{{ params.epochs }}",
        "target_stage": "{{ params.target_stage }}",
        "promotion_min_delta": "{{ params.promotion_min_delta }}",
        "anchor_good_manifest": (
            "{{ params." + anchor_manifest_param + " }}"
        ),
        "anchor_good_max_per_class": (
            "{{ params.anchor_good_max_per_class }}"
        ),
        "reference_eval_manifest": "{{ params.reference_eval_manifest }}",
        "reference_gt_masks_manifest": (
            "{{ params.reference_gt_masks_manifest }}"
        ),
        "progressive_min_defects_for_decision": (
            "{{ params.progressive_min_defects_for_decision }}"
        ),
        "max_good_red_regression": (
            "{{ params.max_good_red_regression }}"
        ),
        "candidate_init_policy": "{{ params.candidate_init_policy }}",
    }


def _define() -> None:
    from airflow.operators.python import ShortCircuitOperator
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

    natural_decision = _collector_task(
        NATURAL_DECISION_TASK_ID,
        "scenario_id",
    )
    natural_gate = ShortCircuitOperator(
        task_id="gate_on_decision",
        python_callable=_should_trigger,
        op_kwargs={"decision_task_id": NATURAL_DECISION_TASK_ID},
    )
    natural_trigger = TriggerDagRunOperator(
        task_id="trigger_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        conf=_lifecycle_conf(
            decision_task_id=NATURAL_DECISION_TASK_ID,
            scenario_param="scenario_id",
            candidate_param="natural_candidate_dataset_version",
            anchor_manifest_param="natural_anchor_good_manifest",
        ),
    )

    drift_decision = _collector_task(
        DRIFT_DECISION_TASK_ID,
        "drift_scenario_id",
    )
    drift_gate = ShortCircuitOperator(
        task_id="gate_on_drift_decision",
        python_callable=_should_trigger,
        op_kwargs={"decision_task_id": DRIFT_DECISION_TASK_ID},
    )
    drift_trigger = TriggerDagRunOperator(
        task_id="trigger_drift_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        conf=_lifecycle_conf(
            decision_task_id=DRIFT_DECISION_TASK_ID,
            scenario_param="drift_scenario_id",
            candidate_param="drift_candidate_dataset_version",
            anchor_manifest_param="drift_anchor_good_manifest",
        ),
    )

    natural_decision >> natural_gate >> natural_trigger
    drift_decision >> drift_gate >> drift_trigger


dag = build_container_dag(
    dag_id="iqa_lifecycle_trigger",
    define=_define,
    schedule="@hourly",
    tags=["iqa", "lifecycle", "trigger"],
    max_active_runs=1,
    catchup=False,
    params={
        "scenario_id": "production_replay_natural",
        "drift_scenario_id": "drift_domain_extension",
        "natural_candidate_dataset_version": "feature_ae_good_v002",
        "drift_candidate_dataset_version": "feature_ae_good_v003",
        "natural_anchor_good_manifest": (
            "data/model_datasets/feature_ae_good_v002.csv"
        ),
        "drift_anchor_good_manifest": (
            "data/model_datasets/feature_ae_good_v003.csv"
        ),
        "roi_window_size": 100,
        "min_natural_conforming": 50,
        "mode": "progressive-train",
        "max_events": 260,
        "lifecycle_interval": 50,
        "max_cycles": 3,
        "epochs": 10,
        "target_stage": "test",
        "promotion_min_delta": 0.0,
        "anchor_good_max_per_class": 256,
        "reference_eval_manifest": "data/validation/validation_set_v001.csv",
        "reference_gt_masks_manifest": (
            "data/validation/validation_gt_masks_v001.csv"
        ),
        "progressive_min_defects_for_decision": 5,
        "max_good_red_regression": 1,
        "candidate_init_policy": "stable_base",
        "image": data_image(),
    },
)
