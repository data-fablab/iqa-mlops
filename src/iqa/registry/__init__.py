"""Model registry skeleton and contracts for IQA."""

from iqa.registry.mlflow_registry import (
    MLflowRegistry,
    ModelRegistryRef,
    register_run_to_model,
    registered_model_name,
)

__all__ = [
    "MLflowRegistry",
    "ModelRegistryRef",
    "registered_model_name",
    "register_run_to_model",
]
