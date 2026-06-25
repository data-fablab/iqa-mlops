"""Canonical Feature-AE model-quality metrics: names, MLflow logging, recall.

Single source of truth for the *business* metrics that (1) appear in the Grafana
dashboards and (2) gate promotion / rollback. The metric names match the
evaluation output (``iqa.training.feature_ae_evaluation.compute_binary_metrics``),
the checkpoint-selection variants, and the documentation
(``docs/modele-feature-ae-iqa.md`` §6): the promotion priority is

    pixel_aupimo_1e-5_1e-3 -> pixel_ap -> image_ap -> image_auroc

Higher is better for all of them, so a *regression* is ``prod - candidate``.
"""

from __future__ import annotations

from typing import Any

# AUPIMO key (low-FPR PIMO integral) as produced by the evaluator / checkpoints.
AUPIMO_KEY = "pixel_aupimo_1e-5_1e-3"

# Business metrics surfaced to dashboards and gates. Order = promotion priority
# (most decisive first). pixel_auroc is logged too but is not a promotion driver.
MODEL_QUALITY_METRIC_KEYS: tuple[str, ...] = (
    AUPIMO_KEY,
    "pixel_ap",
    "image_ap",
    "image_auroc",
)
SUPPLEMENTARY_METRIC_KEYS: tuple[str, ...] = ("pixel_auroc",)
ALL_LOGGED_METRIC_KEYS: tuple[str, ...] = MODEL_QUALITY_METRIC_KEYS + SUPPLEMENTARY_METRIC_KEYS

# All metrics here are "higher is better".
HIGHER_IS_BETTER = True

# MLflow experiment that carries per-model-version quality runs (the exporter and
# the Grafana Postgres datasource both read from it).
MODEL_QUALITY_EXPERIMENT = "iqa-model-quality"

# Stable tag keys so dashboards/gates can filter candidate vs prod by model.
TAG_MODEL_VERSION = "model_version"
TAG_STAGE = "stage"
TAG_METRIC_SOURCE = "metric_source"


def extract_model_quality_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Pull the known business metrics from a raw evaluation metrics dict.

    Skips missing/``None`` values (e.g. pixel metrics when GT masks are absent).
    """
    out: dict[str, float] = {}
    for key in ALL_LOGGED_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def log_model_quality_metrics(
    metrics: dict[str, Any],
    *,
    model_version: str,
    stage: str,
    tracking_uri: str | None = None,
    run_id: str | None = None,
    experiment: str = MODEL_QUALITY_EXPERIMENT,
    extra_tags: dict[str, str] | None = None,
) -> str:
    """Log the business metrics to MLflow so they are queryable for dashboards/gates.

    When ``run_id`` is given the metrics are attached to that existing run
    (e.g. the training run); otherwise a dedicated run is created in
    ``experiment`` named ``{model_version}:{stage}``. Returns the run id used.
    Tags ``model_version`` / ``stage`` / ``metric_source`` make candidate-vs-prod
    filtering possible from Grafana and the exporter.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    business = extract_model_quality_metrics(metrics)
    tags = {
        TAG_MODEL_VERSION: model_version,
        TAG_STAGE: stage,
        TAG_METRIC_SOURCE: "reference_eval",
        **(extra_tags or {}),
    }

    if run_id is not None:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        for name, value in business.items():
            client.log_metric(run_id, name, value)
        for key, value in tags.items():
            client.set_tag(run_id, key, value)
        return run_id

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{model_version}:{stage}") as run:
        mlflow.set_tags(tags)
        for name, value in business.items():
            mlflow.log_metric(name, value)
        return run.info.run_id


def fetch_latest_quality_metrics(
    stage: str,
    *,
    tracking_uri: str | None = None,
    experiment: str = MODEL_QUALITY_EXPERIMENT,
    model_version: str | None = None,
) -> dict[str, float]:
    """Read the latest run's business metrics for a ``stage`` (e.g. ``"prod"``).

    Queries the ``iqa-model-quality`` experiment for runs tagged with ``stage``
    (and optionally ``model_version``), keeps the most recent by ``start_time``,
    and returns its known business metrics. Returns ``{}`` when the experiment or a
    matching run does not exist, so the candidate-vs-prod regression gate can treat
    a missing prod baseline as "not evaluable" rather than crashing.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return {}

    filters = [f"tags.{TAG_STAGE} = '{stage}'"]
    if model_version:
        filters.append(f"tags.{TAG_MODEL_VERSION} = '{model_version}'")
    runs = client.search_runs(
        [found.experiment_id],
        filter_string=" and ".join(filters),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return {}
    metrics = getattr(runs[0].data, "metrics", None) or {}
    return {
        key: float(value)
        for key, value in metrics.items()
        if key in ALL_LOGGED_METRIC_KEYS
    }


def log_per_class_quality_metrics(
    per_class_metrics: dict[str, dict[str, Any]],
    *,
    run_id: str,
    tracking_uri: str | None = None,
) -> dict[str, float]:
    """Log per-source-class business metrics so incremental coverage is visible.

    Flattens ``{source_class: {metric: value}}`` to ``{metric}__{source_class}``
    metric names on the given run (the model-quality run created by
    ``log_model_quality_metrics``). ``None`` values (e.g. pixel metrics without GT
    masks for a class) are skipped. Returns the flat name->value map logged, so the
    incremental coverage of class1/class2/class3 can be charted in Grafana.
    """
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    logged: dict[str, float] = {}
    for source_class in sorted(per_class_metrics):
        class_metrics = per_class_metrics[source_class] or {}
        for key in ALL_LOGGED_METRIC_KEYS:
            value = class_metrics.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            name = f"{key}__{source_class}"
            client.log_metric(run_id, name, numeric)
            logged[name] = numeric
    return logged


__all__ = [
    "ALL_LOGGED_METRIC_KEYS",
    "AUPIMO_KEY",
    "HIGHER_IS_BETTER",
    "MODEL_QUALITY_EXPERIMENT",
    "MODEL_QUALITY_METRIC_KEYS",
    "SUPPLEMENTARY_METRIC_KEYS",
    "TAG_METRIC_SOURCE",
    "TAG_MODEL_VERSION",
    "TAG_STAGE",
    "extract_model_quality_metrics",
    "fetch_latest_quality_metrics",
    "log_model_quality_metrics",
    "log_per_class_quality_metrics",
]
