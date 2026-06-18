"""IQA DVC reproducibility gate DAG: runs the dvc-gate image as a container.

The reproducibility gate is explicit and operator-triggered, but it is still a
container task (ADR 0008). The Airflow image stays lightweight and does not
install the iqa runtime, DVC, git, or repository metadata.

Scope: this DAG validates DVC/MinIO wiring from a dedicated `dvc-gate` image.
`skip_regeneration` defaults to True because the strict metadata regeneration
check requires `.git`, which is deliberately not baked into any runtime image.
"""

from __future__ import annotations

from iqa.dags import build_container_dag, dvc_image, make_container_task


def _define() -> None:
    make_container_task(
        task_id="dvc_reproducibility_check",
        image="{{ params.image }}",
        command=[
            "iqa-check-dvc-reproducibility",
            "--with-network", "{{ params.with_network }}",
            "--skip-regeneration", "{{ params.skip_regeneration }}",
            "--dvc-target", "{{ params.dvc_target }}",
        ],
    )


dag = build_container_dag(
    dag_id="iqa_dvc_reproducibility",
    define=_define,
    schedule=None,
    tags=["iqa", "dvc", "data-lineage"],
    params={
        "with_network": False,
        "skip_regeneration": True,
        "dvc_target": "data/raw/hss-iad.dvc",
        "image": dvc_image(),
    },
)
