"""IQA replay DAG: runs the data image as a container (ADR 0008, issue 12).

The task launches the ``data`` image with ``iqa-run-replay`` via the operator
factory, instead of a BashOperator that assumed ``iqa`` lived in the Airflow
image. Runtime params (scenario_id, plan) are passed as templated argv elements
-- no shell, no quoting.

Replayed events keep their semantics (``event_time``, ``recorded_at``,
``is_simulated``): the boundary validates the plan for the scenario and reports
which of those fields are preserved. Real event emission into the ingestion store
is runtime (data plane), tracked separately.
"""

from __future__ import annotations

from iqa.dags import build_container_dag, data_image, make_container_task


def _define() -> None:
    make_container_task(
        task_id="run_replay",
        image="{{ params.image }}",
        command=[
            "iqa-run-replay",
            "--scenario-id", "{{ params.scenario_id }}",
            "--plan", "{{ params.plan }}",
        ],
    )


dag = build_container_dag(
    dag_id="iqa_replay",
    define=_define,
    schedule=None,
    tags=["iqa", "replay"],
    params={
        "scenario_id": "production_replay_natural",
        "plan": "data/metadata/casting_flux_replay_plan_natural_v003.csv",
        "image": data_image(),
    },
)
