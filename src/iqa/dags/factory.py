"""DAG scaffolding for containerised IQA DAGs (ADR 0008).

Every IQA DAG has the same shape: guard the optional Airflow/provider imports,
open a ``DAG``, add container tasks via :func:`iqa.dags.operators.make_container_task`,
wire their order. This module owns that scaffolding so each DAG module is just
its params, its tasks and their dependencies -- no repeated import/guard ritual.

Like :mod:`iqa.dags.operators`, this stays metier-free (ADR 0008): it imports
Airflow lazily and never the iqa runtime, so the scheduler imports it without
torch or pandas. Each container still carries its own runtime.

The default service images are read from the environment so a deploy can pin a
registry tag without touching the DAGs:

- ``IQA_IMAGE_DATA``  torch-free data runtime image (default ``iqa-data:local``)
- ``IQA_IMAGE_ML``    GPU/torch ml runtime image     (default ``iqa-ml:local``)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime Airflow import.
    from airflow.models.dag import DAG

DEFAULT_DATA_IMAGE = "iqa-data:local"
DEFAULT_ML_IMAGE = "iqa-ml:local"
DEFAULT_START_DATE = datetime(2026, 1, 1)


def data_image() -> str:
    """Service image carrying the torch-free data runtime (override per deploy)."""
    return os.environ.get("IQA_IMAGE_DATA", DEFAULT_DATA_IMAGE)


def ml_image() -> str:
    """Service image carrying the GPU/torch ml runtime (override per deploy)."""
    return os.environ.get("IQA_IMAGE_ML", DEFAULT_ML_IMAGE)


def build_container_dag(
    *,
    dag_id: str,
    define: Callable[[], None],
    params: dict[str, Any],
    tags: list[str],
    schedule: str | None = None,
    catchup: bool = False,
    start_date: datetime = DEFAULT_START_DATE,
    **dag_kwargs: Any,
) -> DAG | None:
    """Open ``dag_id`` and let ``define`` add its container tasks.

    ``define`` is called inside the ``with DAG(...)`` context: it builds the
    tasks (via :func:`make_container_task` and any native Airflow glue) and wires
    their dependencies. Returns the built DAG.

    Returns ``None`` when Airflow or the container provider is absent (e.g. CI
    parsing the module without the Docker/K8s provider): this is the single place
    that knows the DAG is optional, so each DAG module stays a flat declaration
    instead of repeating the import/guard scaffolding.
    """
    try:
        from airflow import DAG
    except ImportError:  # pragma: no cover - Airflow absent (e.g. plain CI).
        return None
    try:
        with DAG(
            dag_id=dag_id,
            schedule=schedule,
            catchup=catchup,
            start_date=start_date,
            tags=tags,
            params=params,
            **dag_kwargs,
        ) as dag:
            define()
        return dag
    except ImportError:  # pragma: no cover - Docker/K8s provider absent (e.g. CI).
        return None


__all__ = ["build_container_dag", "data_image", "ml_image"]
