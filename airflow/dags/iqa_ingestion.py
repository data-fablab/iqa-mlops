"""IQA ingestion DAG: runs the data image as a container (ADR 0008, issue 07).

The task launches the ``data`` image with ``iqa-run-ingestion`` via the operator
factory, instead of a BashOperator that assumed ``iqa`` lived in the Airflow
image. Runtime params (manifest, source, scenario_id) are passed as templated
argv elements -- no shell, no quoting.
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
            dag_id="iqa_ingestion",
            schedule=None,
            catchup=False,
            start_date=datetime(2026, 1, 1),
            tags=["iqa", "ingestion"],
            params={
                "manifest": "data/metadata/casting_piece_events.csv",
                "source": "historical_replay",
                "scenario_id": "raw_ingestion",
                "image": DATA_IMAGE,
            },
        ) as _ingestion_dag:
            make_container_task(
                task_id="run_ingestion",
                image="{{ params.image }}",
                command=[
                    "iqa-run-ingestion",
                    "--manifest", "{{ params.manifest }}",
                    "--source", "{{ params.source }}",
                    "--scenario-id", "{{ params.scenario_id }}",
                ],
            )
        dag = _ingestion_dag
    except ImportError:  # pragma: no cover - Docker/K8s provider absent (e.g. CI).
        dag = None
