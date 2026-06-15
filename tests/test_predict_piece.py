"""Tests for piece prediction aggregation logic."""

from __future__ import annotations

from iqa.inference.contracts import InferenceResult


def _inference_result(
    piece_event_id: str = "piece_001",
    scenario_id: str = "natural",
    score: float = 0.0,
    decision: str = "Vert",
) -> InferenceResult:
    """Create a test inference result."""
    return InferenceResult(
        piece_event_id=piece_event_id,
        scenario_id=scenario_id,
        score=score,
        decision=decision,
        heatmap_uri=None,
        roi_status=None,
        roi_model_version="roi_v001",
        feature_ae_version="ae_v001",
    )


class TestPiecePredictionAggregation:
    """Test piece prediction aggregation from multiple image views."""

    def test_aggregate_single_green_view_is_green(self) -> None:
        """Single green view produces green aggregate."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [_inference_result(score=0.01, decision="Vert")]
        result = aggregate_piece_predictions(views)
        assert result == "Vert"

    def test_aggregate_multiple_green_views_is_green(self) -> None:
        """Multiple green views produce green aggregate."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.02, decision="Vert"),
            _inference_result(score=0.015, decision="Vert"),
        ]
        result = aggregate_piece_predictions(views)
        assert result == "Vert"

    def test_aggregate_orange_view_with_green_is_orange(self) -> None:
        """Orange view among green views produces orange aggregate."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.03, decision="Orange"),
            _inference_result(score=0.02, decision="Vert"),
        ]
        result = aggregate_piece_predictions(views)
        assert result == "Orange"

    def test_aggregate_multiple_orange_views_is_orange(self) -> None:
        """Multiple orange views produce orange aggregate."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(score=0.025, decision="Orange"),
            _inference_result(score=0.035, decision="Orange"),
        ]
        result = aggregate_piece_predictions(views)
        assert result == "Orange"

    def test_aggregate_red_view_overrides_others_is_red(self) -> None:
        """Red view produces red aggregate regardless of other views."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.03, decision="Orange"),
            _inference_result(score=0.06, decision="Rouge"),
        ]
        result = aggregate_piece_predictions(views)
        assert result == "Rouge"

    def test_aggregate_single_red_view_is_red(self) -> None:
        """Single red view produces red aggregate."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [_inference_result(score=0.06, decision="Rouge")]
        result = aggregate_piece_predictions(views)
        assert result == "Rouge"

    def test_aggregate_empty_views_default_green(self) -> None:
        """Empty view list defaults to green (no defects found)."""
        from iqa.inference.piece import aggregate_piece_predictions

        views: list[InferenceResult] = []
        result = aggregate_piece_predictions(views)
        assert result == "Vert"

    def test_aggregate_case_insensitive_decision(self) -> None:
        """Aggregation handles different case variations of decision."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(decision="Vert"),
            _inference_result(decision="vert"),  # Lowercase
            _inference_result(decision="VERT"),  # Uppercase
        ]
        result = aggregate_piece_predictions(views)
        assert result in {"Vert", "vert"}

    def test_aggregate_many_green_one_orange_is_orange(self) -> None:
        """Single orange among many green views produces orange."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.03, decision="Orange"),  # One orange
        ]
        result = aggregate_piece_predictions(views)
        assert result == "Orange"

    def test_aggregate_many_views_one_red_is_red(self) -> None:
        """Single red among many views produces red (fail-safe)."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.03, decision="Orange"),
            _inference_result(score=0.01, decision="Vert"),
            _inference_result(score=0.03, decision="Orange"),
            _inference_result(score=0.06, decision="Rouge"),  # One red
            _inference_result(score=0.01, decision="Vert"),
        ]
        result = aggregate_piece_predictions(views)
        assert result == "Rouge"


class TestPiecePredictionIntegration:
    """Integration tests for piece prediction in API context."""

    def test_aggregate_all_three_statuses(self) -> None:
        """Aggregation covers all three expected statuses."""
        from iqa.inference.piece import aggregate_piece_predictions

        green_views = [_inference_result(decision="Vert")]
        orange_views = [_inference_result(decision="Orange")]
        red_views = [_inference_result(decision="Rouge")]

        assert aggregate_piece_predictions(green_views) == "Vert"
        assert aggregate_piece_predictions(orange_views) == "Orange"
        assert aggregate_piece_predictions(red_views) == "Rouge"

    def test_aggregate_respects_decision_type(self) -> None:
        """Aggregation produces valid Decision type values."""
        from iqa.inference.piece import aggregate_piece_predictions

        views = [
            _inference_result(decision="Vert"),
            _inference_result(decision="Orange"),
            _inference_result(decision="Rouge"),
        ]

        for view in views:
            result = aggregate_piece_predictions([view])
            assert result in {"Vert", "Orange", "Rouge"}
            assert isinstance(result, str)
