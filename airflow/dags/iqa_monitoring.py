"""IQA monitoring DAG: runs the data image as a container (ADR 0008, issue 13).

The hourly task launches the ``data`` image with ``iqa-run-monitoring`` via the
operator factory, instead of a BashOperator that assumed ``iqa`` lived in the
Airflow image. Runtime params (scenario_id, counts, drift, roi_fail_rate,
thresholds config) are passed as templated argv elements -- no shell, no quoting.
``drift_confirmed`` is passed as a value (not a flag) so the argv stays static.

The monitoring thresholds (``configs/monitoring_thresholds.yaml``) are evaluated
inside the container. Pushing the resulting metrics to Prometheus/Grafana is
runtime observability, tracked separately (issue 23).
"""

from __future__ import annotations

try:
    from iqa.dags import build_container_dag, data_image, make_container_task
except ImportError:  # pragma: no cover - iqa package absent from the Airflow image.
    build_container_dag = data_image = make_container_task = None


def _define() -> None:
    make_container_task(
        task_id="evaluate_lifecycle_conditions",
        image="{{ params.image }}",
        command=[
            "iqa-run-monitoring",
            "--scenario-id", "{{ params.scenario_id }}",
            "--conforming-validated-count", "{{ params.conforming_validated_count }}",
            "--drift-confirmed", "{{ params.drift_confirmed }}",
            "--roi-fail-rate", "{{ params.roi_fail_rate }}",
            "--thresholds-config", "{{ params.thresholds_config }}",
        ],
    )


dag = (
    build_container_dag(
        dag_id="iqa_monitoring",
        define=_define,
        schedule="@hourly",
        tags=["iqa", "monitoring"],
        params={
            "scenario_id": "production_replay_natural",
            "conforming_validated_count": 0,
            "drift_confirmed": False,
            "roi_fail_rate": 0.0,
            "thresholds_config": "configs/monitoring_thresholds.yaml",
            "image": data_image(),
        },
    )
    if build_container_dag is not None
    else None
)
