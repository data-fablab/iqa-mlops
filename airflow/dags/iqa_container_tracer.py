"""Tracer DAG: prove a single task runs as a container via the factory (issue 05).

One task, one container (ADR 0008). It uses :func:`iqa.dags.make_container_task`
so the operator choice (Docker today, Kubernetes later) stays in one place. The
container exit code propagates to Airflow, so this DAG doubles as the smoke test
for the wiring landed in issue 06.
"""

from __future__ import annotations

try:
    from iqa.dags import build_container_dag, data_image, make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    build_container_dag = data_image = make_container_task = None


def _define() -> None:
    make_container_task(
        task_id="run_container",
        # exits 0: proves the container ran and the exit code reached Airflow.
        command=["iqa-run-ingestion", "--help"],
        image="{{ params.image }}",
    )


dag = (
    build_container_dag(
        dag_id="iqa_container_tracer",
        define=_define,
        schedule=None,
        tags=["iqa", "tracer"],
        # Default to the data image (no torch, fast to pull); override per-deploy.
        params={"image": data_image()},
    )
    if build_container_dag is not None
    else None
)
