"""Tracer DAG: prove a single task runs as a container via the factory (issue 05).

One task, one container (ADR 0008). It uses :func:`iqa.dags.operators.make_container_task`
so the operator choice (Docker today, Kubernetes later) stays in one place. The
container exit code propagates to Airflow, so this DAG doubles as the smoke test
for the wiring landed in issue 06.
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


# Default to the data image (no torch, fast to pull); override per-deploy.
TRACER_IMAGE = os.environ.get("IQA_IMAGE_DATA", "iqa-data:local")


dag = None
if DAG is not None and make_container_task is not None:
    try:
        with DAG(
            dag_id="iqa_container_tracer",
            schedule=None,
            catchup=False,
            start_date=datetime(2026, 1, 1),
            tags=["iqa", "tracer"],
            params={"image": TRACER_IMAGE},
        ) as _tracer_dag:
            make_container_task(
                task_id="run_container",
                image="{{ params.image }}",
                # exits 0: proves the container ran and the exit code reached Airflow.
                command=["iqa-run-ingestion", "--help"],
            )
        dag = _tracer_dag
    except ImportError:  # pragma: no cover - Docker/K8s provider absent (e.g. CI).
        dag = None
