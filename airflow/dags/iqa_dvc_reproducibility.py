"""IQA DVC reproducibility gate DAG: runs the dvc-gate image as a container (ADR 0008).

The reproducibility gate (``iqa-check-dvc-reproducibility``) runs as a container
on the dedicated ``dvc-gate`` image via the operator factory, instead of a
BashOperator that assumed ``iqa`` (and the dvc/git toolchain + the repo's
``.dvc``/``dvc.yaml``) lived in the Airflow image. The boolean params are passed
as templated argv *values* (not Jinja-conditional flags) so the argv stays
static and shell-free, like ``iqa_monitoring``.

Reproducibility is a repo-level check: it lists the DVC remote, optionally
pulls/pushes the DVC-tracked source against MinIO. The gate image therefore ships
``dvc[s3]`` + git and the repo's ``.dvc``/``dvc.yaml``/``*.dvc`` (see the
``dvc-gate`` Dockerfile target); MinIO credentials reach it through the env
allowlist (ADR 0008 / KEN06).

Scope (image-friendly): the container runs the DVC/MinIO checks only.
``skip_regeneration`` defaults to ``True`` because the deterministic-regeneration
check needs ``git diff`` against the repo history (``.git``), which we
deliberately keep out of any image; that determinism gate stays in CI.
"""

from __future__ import annotations

try:
    from iqa.dags import build_container_dag, dvc_image, make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    build_container_dag = dvc_image = make_container_task = None


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
            "skip_regeneration": True,
            "dvc_target": "data/raw/hss-iad.dvc",
            "image": dvc_image(),
        },
    )
    if build_container_dag is not None
    else None
)
