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

import os
from datetime import datetime

try:
    from airflow import DAG
except ImportError:  # pragma: no cover - lets CI import the module without Airflow.
    DAG = None

try:
    from iqa.dags.operators import make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    make_container_task = None


DATA_IMAGE = os.environ.get("IQA_IMAGE_DATA", "iqa-data:local")


dag = None
if DAG is not None and make_container_task is not None:
    try:
        with DAG(
            dag_id="iqa_replay",
            schedule=None,
            catchup=False,
            start_date=datetime(2026, 1, 1),
            tags=["iqa", "replay"],
            params={
                "scenario_id": "production_replay_natural",
                "plan": "data/metadata/casting_flux_replay_plan_natural.csv",
                "image": DATA_IMAGE,
            },
        ) as _replay_dag:
            make_container_task(
                task_id="run_replay",
                image="{{ params.image }}",
                command=[
                    "iqa-run-replay",
                    "--scenario-id", "{{ params.scenario_id }}",
                    "--plan", "{{ params.plan }}",
                ],
            )
        dag = _replay_dag
    except ImportError:  # pragma: no cover - Docker/K8s provider absent (e.g. CI).
        dag = None
