"""Contract tests for MLflow Registry skeleton (IQA1_KEN08)."""
from __future__ import annotations

import pytest

from iqa.registry.mlflow_registry import MLflowRegistry


def test_get_model_returns_prod_by_default() -> None:
    registry = MLflowRegistry()
    model = registry.get_model("feature_ae__production_replay_natural")

    assert model is not None
    assert model.scenario_id == "production_replay_natural"
    assert model.stage == "prod"
    assert model.registered_model_name == "feature_ae__production_replay_natural"


def test_get_model_returns_candidate_when_requested() -> None:
    registry = MLflowRegistry()
    model = registry.get_model("feature_ae__production_replay_natural", stage="candidate")

    assert model is not None
    assert model.stage == "candidate"


def test_get_model_returns_test_when_requested() -> None:
    registry = MLflowRegistry()
    model = registry.get_model("feature_ae__production_replay_natural", stage="test")

    assert model is not None
    assert model.stage == "test"


def test_get_model_returns_archived_when_requested() -> None:
    registry = MLflowRegistry()
    model = registry.get_model("feature_ae__production_replay_natural", stage="archived")

    assert model is not None
    assert model.stage == "archived"


def test_list_models_returns_all_stages() -> None:
    registry = MLflowRegistry()
    models = registry.list_models("feature_ae__production_replay_natural")

    assert len(models) == 4
    assert "prod" in models
    assert "candidate" in models
    assert "test" in models
    assert "archived" in models
    assert models["prod"].stage == "prod"
    assert models["candidate"].stage == "candidate"
    assert models["test"].stage == "test"
    assert models["archived"].stage == "archived"


def test_list_scenarios_returns_all_registered_models() -> None:
    registry = MLflowRegistry()
    scenarios = registry.list_scenarios()

    assert len(scenarios) >= 2
    assert "feature_ae__production_replay_natural" in scenarios
    assert "roi__surface_defects" in scenarios


def test_only_four_lifecycle_stages_exist() -> None:
    registry = MLflowRegistry()
    models = registry.list_models("feature_ae__production_replay_natural")
    stages = set(models.keys())

    assert stages == {"candidate", "test", "prod", "archived"}


def test_invalid_stage_raises_error() -> None:
    registry = MLflowRegistry()

    with pytest.raises(ValueError, match="Unknown stage"):
        registry.get_model("feature_ae__production_replay_natural", stage="invalid")
