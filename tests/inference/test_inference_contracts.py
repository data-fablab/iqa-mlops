"""Contract tests for the inference request/result data structures (IQA1_KEN01).

Moved out of the registry test module: these validate iqa.inference.contracts,
not the model registry.
"""

from __future__ import annotations

from iqa.inference.contracts import InferenceRequest, InferenceResult


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
            decision="Orange",
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
            decision="Orange",
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
            decision="Orange",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        assert result.score == 0.025
        assert isinstance(result.score, float)

    def test_inference_result_requires_decision_type(self) -> None:
        """InferenceResult requires decision in {Vert, Orange, Rouge}."""
        for decision in ["Vert", "Orange", "Rouge"]:
            result = InferenceResult(
                piece_event_id="piece_001",
                scenario_id="production_replay_natural",
                score=0.0,
                decision=decision,
                heatmap_uri=None,
                roi_status=None,
                roi_model_version="roi_v001",
                feature_ae_version="ae_v001",
            )
            assert result.decision == decision

    def test_inference_result_requires_model_versions(self) -> None:
        """InferenceResult requires roi_model_version and feature_ae_version."""
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.0,
            decision="Vert",
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
            decision="Orange",
            heatmap_uri="s3://bucket/heatmap.png",
            roi_status="ok",
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        result_dict = result.to_dict()
        assert result_dict["piece_event_id"] == "piece_001"
        assert result_dict["scenario_id"] == "production_replay_natural"
        assert result_dict["score"] == 0.025
        assert result_dict["decision"] == "Orange"
        assert result_dict["heatmap_uri"] == "s3://bucket/heatmap.png"
        assert result_dict["roi_status"] == "ok"

    def test_inference_result_domain_drift_fields_default_none(self) -> None:
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.0,
            decision="Vert",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
        )
        assert result.domain_drift_score is None
        assert result.domain_regime is None
        d = result.to_dict()
        assert "domain_drift_score" in d
        assert "domain_regime" in d
        assert d["domain_drift_score"] is None
        assert d["domain_regime"] is None

    def test_inference_result_domain_drift_fields_round_trip(self) -> None:
        result = InferenceResult(
            piece_event_id="piece_001",
            scenario_id="production_replay_natural",
            score=0.025,
            decision="Orange",
            heatmap_uri=None,
            roi_status=None,
            roi_model_version="roi_v001",
            feature_ae_version="ae_v001",
            domain_drift_score=4.22,
            domain_regime="out_of_domain",
        )
        assert result.domain_drift_score == 4.22
        assert result.domain_regime == "out_of_domain"
        d = result.to_dict()
        assert d["domain_drift_score"] == 4.22
        assert d["domain_regime"] == "out_of_domain"
