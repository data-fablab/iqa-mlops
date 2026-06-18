"""IQA DVC reproducibility gate DAG: runs the data image as a container (ADR 0008).

The reproducibility gate (``iqa-check-dvc-reproducibility``) runs as a container
on the ``data`` image via the operator factory, instead of a BashOperator that
assumed ``iqa`` (and the dvc/git toolchain) lived in the Airflow image. The
boolean params are passed as templated argv *values* (not Jinja-conditional
flags) so the argv stays static and shell-free, like ``iqa_monitoring``.

Reproducibility is a repo-level check: it lists the DVC remote, optionally
pulls/pushes the DVC-tracked source against MinIO, and asserts deterministic
metadata regeneration leaves no Git diff. The task container therefore needs the
dvc/git toolchain and read access to the repo (``.dvc``/``.git``/``dvc.yaml``) --
which the light ``data`` image deliberately excludes (``.dockerignore``). Wiring
that runtime (dvc extra + read-only repo mount, or a dedicated gate image) is an
open decision tracked separately; this module only owns the orchestration shape.
"""

from __future__ import annotations

try:
    from iqa.dags import build_container_dag, data_image, make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    build_container_dag = data_image = make_container_task = None


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


dag = (
    build_container_dag(
        dag_id="iqa_dvc_reproducibility",
        define=_define,
        schedule=None,
        tags=["iqa", "dvc", "data-lineage"],
        params={
            "with_network": False,
            "skip_regeneration": False,
            "dvc_target": "data/raw/hss-iad.dvc",
            "image": data_image(),
        },
    )
    if build_container_dag is not None
    else None
)
