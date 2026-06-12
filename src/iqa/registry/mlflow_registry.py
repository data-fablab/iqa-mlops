"""MLflow Registry naming contracts.

MLflow is the source of truth for promotion and rollback. MinIO stores model
artifacts but does not decide which model is production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


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


__all__ = ["ModelRegistryRef", "registered_model_name"]
