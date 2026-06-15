"""Model rollback: restore previous_prod, archive faulty version."""

from __future__ import annotations

from typing import Any


def save_previous_prod_before_promotion(
    registered_model_name: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Save current prod version as previous_prod in MLflow.

    This must be called BEFORE promoting a new model. It preserves the current
    production model so we can rollback if needed.

    Args:
        registered_model_name: Name of the registered model
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with:
        - success: bool
        - previous_prod_version: str (the prod version that was saved)
        - registered_model_name: str
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for save_previous_prod_before_promotion")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    try:
        versions = client.get_latest_versions(registered_model_name, stages=["prod"])
        if not versions:
            raise ValueError(f"No prod model found for {registered_model_name}")

        prod_version = versions[0]
        previous_prod_version = str(prod_version.version)

        return {
            "success": True,
            "previous_prod_version": previous_prod_version,
            "registered_model_name": registered_model_name,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "registered_model_name": registered_model_name,
        }


def get_previous_prod(
    registered_model_name: str,
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Get the saved previous_prod version for a model.

    Args:
        registered_model_name: Name of the registered model
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with:
        - version: str (the previous_prod version)
        - registered_model_name: str
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for get_previous_prod")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    try:
        # Get the alias "previous_prod" if it exists
        model_version = client.get_model_version_by_alias(
            registered_model_name, "previous_prod"
        )
        return {
            "version": str(model_version.version),
            "registered_model_name": registered_model_name,
        }
    except Exception as e:
        raise ValueError(
            f"No previous_prod version found for {registered_model_name}: {e}"
        )


def rollback_model(
    registered_model_name: str,
    faulty_version: str,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Rollback from faulty version: restore previous_prod to prod, archive faulty.

    Args:
        registered_model_name: Name of the registered model
        faulty_version: Version to rollback from (will be archived)
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with:
        - success: bool
        - registered_model_name: str
        - previous_prod_version: str (restored to prod)
        - faulty_version_archived: str
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for rollback_model")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    try:
        # Get the previous_prod version
        previous_prod_info = get_previous_prod(registered_model_name, tracking_uri)
        previous_prod_version = previous_prod_info["version"]

        # Transition previous_prod back to prod
        client.transition_model_version_stage(
            name=registered_model_name,
            version=previous_prod_version,
            stage="prod",
        )

        # Archive the faulty version
        client.transition_model_version_stage(
            name=registered_model_name,
            version=str(faulty_version),
            stage="archived",
        )

        return {
            "success": True,
            "registered_model_name": registered_model_name,
            "previous_prod_version": previous_prod_version,
            "faulty_version_archived": str(faulty_version),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "registered_model_name": registered_model_name,
        }


__all__ = [
    "save_previous_prod_before_promotion",
    "get_previous_prod",
    "rollback_model",
]
