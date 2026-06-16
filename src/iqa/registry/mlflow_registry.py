"""MLflow Registry skeleton managing model lifecycle states.

MLflow is the source of truth for promotion and rollback. MinIO stores model
artifacts but does not decide which model is production.

States: candidate, test, prod, archived (per scenario_id)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass(frozen=True)
class ModelRegistryRef:
    scenario_id: str
    registered_model_name: str
    stage: str = "prod"
    source_of_truth: str = "mlflow_registry"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def registered_model_name(scenario_id: str, *, base_name: str = "feature_ae") -> str:
    if not scenario_id:
        raise ValueError("scenario_id is required for IQA registered model names.")
    return f"{base_name}__{scenario_id}"


class MLflowRegistry:
    """Registry skeleton for model lifecycle management by scenario_id."""

    # Skeleton dataset: scenarios with their states
    _REGISTRY = {
        "feature_ae__production_replay_natural": {
            "prod": ModelRegistryRef(
                scenario_id="production_replay_natural",
                registered_model_name="feature_ae__production_replay_natural",
                stage="prod",
            ),
            "candidate": ModelRegistryRef(
                scenario_id="production_replay_natural",
                registered_model_name="feature_ae__production_replay_natural",
                stage="candidate",
            ),
            "test": ModelRegistryRef(
                scenario_id="production_replay_natural",
                registered_model_name="feature_ae__production_replay_natural",
                stage="test",
            ),
            "archived": ModelRegistryRef(
                scenario_id="production_replay_natural",
                registered_model_name="feature_ae__production_replay_natural",
                stage="archived",
            ),
        },
        "roi__surface_defects": {
            "prod": ModelRegistryRef(
                scenario_id="surface_defects",
                registered_model_name="roi__surface_defects",
                stage="prod",
            ),
            "candidate": ModelRegistryRef(
                scenario_id="surface_defects",
                registered_model_name="roi__surface_defects",
                stage="candidate",
            ),
            "test": ModelRegistryRef(
                scenario_id="surface_defects",
                registered_model_name="roi__surface_defects",
                stage="test",
            ),
            "archived": ModelRegistryRef(
                scenario_id="surface_defects",
                registered_model_name="roi__surface_defects",
                stage="archived",
            ),
        },
    }

    def get_model(self, registered_model_name: str, *, stage: str = "prod") -> ModelRegistryRef:
        """Get a model by registered_model_name and stage."""
        if registered_model_name not in self._REGISTRY:
            raise ValueError(f"Unknown registered model: {registered_model_name}")
        if stage not in self._REGISTRY[registered_model_name]:
            raise ValueError(f"Unknown stage: {stage}")
        return self._REGISTRY[registered_model_name][stage]

    def list_models(self, registered_model_name: str) -> dict[str, ModelRegistryRef]:
        """List all models (across all stages) for a registered_model_name."""
        if registered_model_name not in self._REGISTRY:
            raise ValueError(f"Unknown registered model: {registered_model_name}")
        return self._REGISTRY[registered_model_name]

    def list_scenarios(self) -> list[str]:
        """List all registered_model_names."""
        return list(self._REGISTRY.keys())


def register_run_to_model(
    run_id: str,
    scenario_id: str,
    stage: str = "candidate",
    model_name_base: str = "feature_ae",
    tracking_uri: str | None = None,
) -> dict[str, str]:
    """Register an MLflow run as a model version in a registered model.

    Args:
        run_id: MLflow run ID to register
        scenario_id: Scenario identifier (used to determine registered model name)
        stage: MLflow stage (candidate, test, prod, archived)
        model_name_base: Base name for the registered model (default: feature_ae)
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        Dict with keys:
        - registered_model_name: str
        - version: str (version number)
        - stage: str (the stage that was set)
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow is required for register_run_to_model")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Determine registered model name
    model_name = registered_model_name(scenario_id, base_name=model_name_base)

    run = client.get_run(run_id)
    model_source = f"{run.info.artifact_uri.rstrip('/')}/model"

    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException as e:
        if "already exists" not in str(e):
            raise

    model_version = client.create_model_version(
        name=model_name,
        source=model_source,
        run_id=run_id,
    )
    if model_version is None:
        raise RuntimeError(f"Failed to register model {model_name} from run {run_id}")

    version_str = str(model_version.version)

    # Stages are deprecated; mark the version with an alias of the same name.
    if stage and stage != "None":
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=stage,
                version=version_str,
            )
        except Exception:
            # Aliasing might fail if the version is already aliased.
            pass

    return {
        "registered_model_name": model_name,
        "version": version_str,
        "stage": stage,
    }


__all__ = [
    "ModelRegistryRef",
    "registered_model_name",
    "MLflowRegistry",
    "register_run_to_model",
]
