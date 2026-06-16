"""Shared construction of MLflow clients.

Centralizes the "import mlflow, set tracking URI, build MlflowClient" boilerplate
that promotion/rollback workflows would otherwise duplicate in every function.
"""

from __future__ import annotations

from typing import Any


def get_mlflow_client(
    tracking_uri: str | None = None,
    *,
    required_for: str | None = None,
) -> Any:
    """Return an ``MlflowClient``, setting the tracking URI when provided.

    Args:
        tracking_uri: MLflow tracking URI (optional).
        required_for: Operation name used in the ImportError message when MLflow
            is not installed (e.g. "model promotion").

    Returns:
        A configured ``mlflow.tracking.MlflowClient``.

    Raises:
        ImportError: If MLflow is not installed.
    """
    try:
        import mlflow
    except ImportError as exc:
        target = required_for or "this operation"
        raise ImportError(f"MLflow is required for {target}") from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


__all__ = ["get_mlflow_client"]
