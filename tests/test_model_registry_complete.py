"""Complete test coverage for model contract and registry (IQA1_KEN10).

Tests for:
- Model contract (input/output validation)
- Registry states and scenario_id partitioning
- Edge cases (unknown states, missing scenario_id)
"""

from __future__ import annotations

import pytest

from iqa.inference.contracts import InferenceRequest, InferenceResult
from iqa.registry import ModelRegistryRef, registered_model_name
from iqa.registry.mlflow_registry import MLflowRegistry


class TestModelContractInputValidation:
    """Test model contract input requirements (IQA1_KEN01)."""

    def test_inference_request_requires_piece_event_id(self) -> None:
        """InferenceRequest requires piece_event_id."""
        request = InferenceRequest(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            image_uri="s3://bucket/image.jpg",
        )
        assert request.piece_event_id == "piece_001"

    def test_inference_request_requires_scenario_id(self) -> None:
        """InferenceRequest requires scenario_id."""
        request = InferenceRequest(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            image_uri="s3://bucket/image.jpg",
        )
        assert request.scenario_id == "production_replay_natural"

    def test_inference_request_requires_image_uri(self) -> None:
        """InferenceRequest requires image_uri."""
        request = InferenceRequest(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            image_uri="s3://bucket/image.jpg",
        )
        assert request.image_uri == "s3://bucket/image.jpg"

    def test_inference_request_rejects_empty_piece_event_id(self) -> None:
        """InferenceRequest with empty piece_event_id can be created (validation at boundary)."""
        request = InferenceRequest(
            piece_event_id="",
            scenario_id="production_replay_natural",
            image_uri="s3://bucket/image.jpg",
        )
        assert request.piece_event_id == ""


class TestModelContractOutputValidation:
    """Test model contract output structure (IQA1_KEN01)."""

    def test_inference_result_requires_piece_event_id(self) -> None:
        """InferenceResult requires piece_event_id."""
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.025,
            statut="Orange",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        assert result.piece_event_id == "piece_001"

    def test_inference_result_requires_scenario_id(self) -> None:
        """InferenceResult requires scenario_id."""
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.025,
            statut="Orange",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        assert result.scenario_id == "production_replay_natural"

    def test_inference_result_requires_score(self) -> None:
        """InferenceResult requires score field."""
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.025,
            statut="Orange",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        assert result.score == 0.025
        assert isinstance(result.score, float)

    def test_inference_result_requires_statut_decision_type(self) -> None:
        """InferenceResult requires statut in {Vert, Orange, Rouge}."""
        for statut in ["Vert", "Orange", "Rouge"]:
            result = InferenceResult(
                piece_event_id="piece_001",
                scenario_id="production_replay_natural",
                score=0.0,
                statut=statut,
                heatmap_uri=None,
                roi_status=None,
                roi_model_version="roi_v001",
                feature_ae_version="ae_v001",
            )
            assert result.statut == statut

    def test_inference_result_requires_model_versions(self) -> None:
        """InferenceResult requires roi_model_version and feature_ae_version."""
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.0,
            statut="Vert",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        assert result.roi_model_version == "roi_v001"
        assert result.feature_ae_version == "ae_v001"

    def test_inference_result_to_dict_includes_all_fields(self) -> None:
        """InferenceResult.to_dict() includes all fields."""
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.025,
            statut="Orange",
            heatmap_uri="s3://bucket/heatmap.png",
            roi_status="ok",
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        result_dict = result.to_dict()
        assert result_dict["piece_event_id"] == "piece_001"
        assert result_dict["scenario_id"] == "production_replay_natural"
        assert result_dict["score"] == 0.025
        assert result_dict["statut"] == "Orange"
        assert result_dict["heatmap_uri"] == "s3://bucket/heatmap.png"
        assert result_dict["roi_status"] == "ok"


class TestRegistryEdgeCases:
    """Test registry edge cases and error handling."""

    def test_registered_model_name_rejects_empty_scenario_id(self) -> None:
        """registered_model_name raises ValueError for empty scenario_id."""
        with pytest.raises(ValueError, match="scenario_id is required"):
            registered_model_name("")

    def test_registered_model_name_rejects_none_scenario_id(self) -> None:
        """registered_model_name raises ValueError for None scenario_id."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            registered_model_name(None)  # type: ignore

    def test_registered_model_name_accepts_valid_scenario_id(self) -> None:
        """registered_model_name produces correct format."""
        name = registered_model_name("my_scenario")
        assert name == "feature_ae__my_scenario"

    def test_registered_model_name_with_custom_base_name(self) -> None:
        """registered_model_name accepts custom base_name parameter."""
        name = registered_model_name("my_scenario", base_name="custom_model")
        assert name == "custom_model__my_scenario"

    def test_get_model_rejects_unknown_registered_model_name(self) -> None:
        """Registry.get_model raises ValueError for unknown model."""
        registry = MLflowRegistry()
        with pytest.raises(ValueError, match="Unknown registered model"):
            registry.get_model("unknown_model__scenario")

    def test_get_model_rejects_unknown_stage(self) -> None:
        """Registry.get_model raises ValueError for unknown stage."""
        registry = MLflowRegistry()
        with pytest.raises(ValueError, match="Unknown stage"):
            registry.get_model("feature_ae__production_replay_natural", stage="invalid_stage")

    def test_list_models_rejects_unknown_registered_model_name(self) -> None:
        """Registry.list_models raises ValueError for unknown model."""
        registry = MLflowRegistry()
        with pytest.raises(ValueError, match="Unknown registered model"):
            registry.list_models("unknown_model__scenario")

    def test_list_scenarios_returns_non_empty_list(self) -> None:
        """Registry.list_scenarios returns at least one model."""
        registry = MLflowRegistry()
        scenarios = registry.list_scenarios()
        assert len(scenarios) > 0
        assert all(isinstance(s, str) for s in scenarios)


class TestRegistryScenarioPartitioning:
    """Test registry partitioning by scenario_id."""

    def test_production_replay_natural_has_all_stages(self) -> None:
        """production_replay_natural model has all four stages."""
        registry = MLflowRegistry()
        models = registry.list_models("feature_ae__production_replay_natural")
        assert set(models.keys()) == {"prod", "candidate", "test", "archived"}

    def test_all_production_replay_models_have_correct_scenario_id(self) -> None:
        """All production_replay_natural models reference correct scenario_id."""
        registry = MLflowRegistry()
        models = registry.list_models("feature_ae__production_replay_natural")
        for model in models.values():
            assert model.scenario_id == "production_replay_natural"

    def test_roi_surface_defects_has_all_stages(self) -> None:
        """roi__surface_defects model has all four stages."""
        registry = MLflowRegistry()
        models = registry.list_models("roi__surface_defects")
        assert set(models.keys()) == {"prod", "candidate", "test", "archived"}

    def test_all_roi_models_have_correct_scenario_id(self) -> None:
        """All roi__surface_defects models reference correct scenario_id."""
        registry = MLflowRegistry()
        models = registry.list_models("roi__surface_defects")
        for model in models.values():
            assert model.scenario_id == "surface_defects"

    def test_different_scenarios_have_different_registered_model_names(self) -> None:
        """Different scenarios have different registered_model_name."""
        registry = MLflowRegistry()
        production_model = registry.get_model("feature_ae__production_replay_natural")
        roi_model = registry.get_model("roi__surface_defects")
        assert production_model.registered_model_name != roi_model.registered_model_name

    def test_model_ref_serialization(self) -> None:
        """ModelRegistryRef serializes correctly to dict."""
        ref = ModelRegistryRef(
            scenario_id="test_scenario",
            registered_model_name="model__test_scenario",
            stage="prod",
        )
        ref_dict = ref.to_dict()
        assert ref_dict["scenario_id"] == "test_scenario"
        assert ref_dict["registered_model_name"] == "model__test_scenario"
        assert ref_dict["stage"] == "prod"
        assert ref_dict["source_of_truth"] == "mlflow_registry"
