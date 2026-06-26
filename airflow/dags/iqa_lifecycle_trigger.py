"""Poll durable lifecycle signals and trigger the Feature-AE lifecycle.

Every hour, two independent branches collect PostgreSQL-backed signals:

- natural replay: new conforming and train-eligible ``oracle_gt`` feedback;
- drift replay: the latest unconsumed versioned drift observation.

The data-image container evaluates and journals each decision. Native Airflow
glue only parses the returned JSON, gates the branch and triggers
``iqa_lifecycle``. ``max_active_runs=1`` prevents overlapping polling runs.
"""

from __future__ import annotations

import json

from iqa.dags import build_container_dag, data_image, make_container_task


NATURAL_DECISION_TASK_ID = "evaluate_decision"
DRIFT_DECISION_TASK_ID = "evaluate_drift_decision"
NATURAL_CONF_TASK_ID = "build_trigger_conf"
DRIFT_CONF_TASK_ID = "build_drift_trigger_conf"
LIFECYCLE_DAG_ID = "iqa_lifecycle"


def _decision_payload(task_id: str, ti=None) -> dict:
    payload = ti.xcom_pull(task_ids=task_id)
    if payload is None:
        return {}
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    return payload


def _should_trigger(decision_task_id: str, ti=None, **_context) -> bool:
    payload = _decision_payload(decision_task_id, ti)
    return bool(payload.get("trigger_lifecycle", False))


def _build_lifecycle_conf(
    decision_task_id: str,
    ti=None,
    params=None,
    **_context,
) -> dict:
    payload = _decision_payload(decision_task_id, ti)
    signal = payload.get("signal") or {}
    decision = payload.get("lifecycle_decision") or {}
    params = params or {}

    return {
        "scenario_id": signal.get("scenario_id"),
        "conforming_validated_count": signal.get(
            "conforming_validated_count",
            0,
        ),
        "drift_confirmed": signal.get("drift_confirmed", False),
        "roi_fail_rate": signal.get("roi_fail_rate", 0.0),
        "lifecycle_trigger_event_id": payload.get(
            "lifecycle_trigger_event_id"
        ),
        "trigger_reason": decision.get("trigger_reason"),
        "candidate_dataset_version": decision.get(
            "candidate_dataset_version"
        ),
        "watermark": payload.get("watermark"),
        "image_root": params.get("image_root"),
        "mode": params.get("mode"),
        "max_events": params.get("max_events"),
        "lifecycle_interval": params.get("lifecycle_interval"),
        "max_cycles": params.get("max_cycles"),
        "epochs": params.get("epochs"),
        "target_stage": params.get("target_stage"),
        "promotion_min_delta": params.get("promotion_min_delta"),
        "anchor_good_manifest": params.get("anchor_good_manifest"),
        "anchor_good_max_per_class": params.get(
            "anchor_good_max_per_class"
        ),
        "reference_eval_manifest": params.get(
            "reference_eval_manifest"
        ),
        "reference_gt_masks_manifest": params.get(
            "reference_gt_masks_manifest"
        ),
        "progressive_min_defects_for_decision": params.get(
            "progressive_min_defects_for_decision"
        ),
        "max_good_red_regression": params.get(
            "max_good_red_regression"
        ),
        "candidate_init_policy": params.get("candidate_init_policy"),
    }


def _trigger_conf(task_id: str) -> dict:
    expression = f"ti.xcom_pull(task_ids='{task_id}')"
    fields = (
        "scenario_id",
        "conforming_validated_count",
        "drift_confirmed",
        "roi_fail_rate",
        "lifecycle_trigger_event_id",
        "trigger_reason",
        "candidate_dataset_version",
        "watermark",
        "image_root",
        "mode",
        "max_events",
        "lifecycle_interval",
        "max_cycles",
        "epochs",
        "target_stage",
        "promotion_min_delta",
        "anchor_good_manifest",
        "anchor_good_max_per_class",
        "reference_eval_manifest",
        "reference_gt_masks_manifest",
        "progressive_min_defects_for_decision",
        "max_good_red_regression",
        "candidate_init_policy",
    )
    return {
        field: "{{ " + expression + "['" + field + "'] }}"
        for field in fields
    }


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


def _define() -> None:
    from airflow.operators.python import PythonOperator, ShortCircuitOperator
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
    natural_conf = PythonOperator(
        task_id=NATURAL_CONF_TASK_ID,
        python_callable=_build_lifecycle_conf,
        op_kwargs={"decision_task_id": NATURAL_DECISION_TASK_ID},
    )
    natural_trigger = TriggerDagRunOperator(
        task_id="trigger_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        conf=_trigger_conf(NATURAL_CONF_TASK_ID),
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
    drift_conf = PythonOperator(
        task_id=DRIFT_CONF_TASK_ID,
        python_callable=_build_lifecycle_conf,
        op_kwargs={"decision_task_id": DRIFT_DECISION_TASK_ID},
    )
    drift_trigger = TriggerDagRunOperator(
        task_id="trigger_drift_lifecycle",
        trigger_dag_id=LIFECYCLE_DAG_ID,
        conf=_trigger_conf(DRIFT_CONF_TASK_ID),
    )

    natural_decision >> natural_gate >> natural_conf >> natural_trigger
    drift_decision >> drift_gate >> drift_conf >> drift_trigger


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
        "roi_window_size": 100,
        "min_natural_conforming": 50,
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
        "reference_eval_manifest": "data/validation/validation_set_v001.csv",
        "reference_gt_masks_manifest": "data/validation/validation_gt_masks_v001.csv",
        "progressive_min_defects_for_decision": 5,
        "max_good_red_regression": 1,
        "candidate_init_policy": "stable_base",
        "image": data_image(),
    },
)
