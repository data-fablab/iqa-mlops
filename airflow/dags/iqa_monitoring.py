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

from iqa.dags import build_container_dag, data_image, make_container_task


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
            "--source-domain", "{{ params.source_domain }}",
            "--window-events", "{{ params.window_events }}",
            "--domain-ratio", "{{ params.domain_ratio }}",
            "--alert-rate", "{{ params.alert_rate }}",
            "--red-rate", "{{ params.red_rate }}",
            "--unexpected-red-rate", "{{ params.unexpected_red_rate }}",
            "--oracle-fn-rate", "{{ params.oracle_fn_rate }}",
            "--critical-window-count", "{{ params.critical_window_count }}",
            "--api-url", "{{ params.api_url }}",
            "--thresholds-config", "{{ params.thresholds_config }}",
        ],
    )


dag = build_container_dag(
    dag_id="iqa_monitoring",
    define=_define,
    schedule="@hourly",
    tags=["iqa", "monitoring"],
    params={
        "scenario_id": "production_replay_natural",
        "conforming_validated_count": 0,
        "drift_confirmed": False,
        "roi_fail_rate": 0.0,
        "source_domain": "piece_a_p4",
        "window_events": 0,
        "domain_ratio": 0.0,
        "alert_rate": 0.0,
        "red_rate": 0.0,
        "unexpected_red_rate": 0.0,
        "oracle_fn_rate": 0.0,
        "critical_window_count": 0,
        "api_url": "http://iqa-api:8000",
        "thresholds_config": "configs/monitoring_thresholds.yaml",
        "image": data_image(),
    },
)
