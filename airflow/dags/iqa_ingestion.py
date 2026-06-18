"""IQA ingestion DAG: runs the data image as a container (ADR 0008, issue 07).

The task launches the ``data`` image with ``iqa-run-ingestion`` via the operator
factory, instead of a BashOperator that assumed ``iqa`` lived in the Airflow
image. Runtime params (manifest, source, scenario_id) are passed as templated
argv elements -- no shell, no quoting. The import/guard scaffolding lives in
:func:`iqa.dags.build_container_dag`.
"""

from __future__ import annotations

try:
    from iqa.dags import build_container_dag, data_image, make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    build_container_dag = data_image = make_container_task = None


def _define() -> None:
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


dag = (
    build_container_dag(
        dag_id="iqa_ingestion",
        define=_define,
        schedule=None,
        tags=["iqa", "ingestion"],
        params={
            "manifest": "data/metadata/casting_piece_events.csv",
            "source": "historical_replay",
            "scenario_id": "raw_ingestion",
            "image": data_image(),
        },
    )
    if build_container_dag is not None
    else None
)
