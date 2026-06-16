"""Tests for piece prediction aggregation logic.

Aggregation rule: the worst (most severe) status wins (Rouge > Orange > Vert).
"""

from __future__ import annotations

from iqa.inference.contracts import InferenceResult
from iqa.inference.piece import aggregate_piece_predictions


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

    def test_aggregate_empty_views_default_green(self) -> None:
        """Empty view list defaults to green (no defects found)."""
        result = aggregate_piece_predictions([])
        assert result == "Vert"

    def test_aggregate_all_green_views_is_green(self) -> None:
        """Only green views produce a green aggregate."""
        views = [
            _inference_result(decision="Vert"),
            _inference_result(decision="Vert"),
            _inference_result(decision="Vert"),
        ]
        assert aggregate_piece_predictions(views) == "Vert"

    def test_aggregate_orange_presence_is_orange(self) -> None:
        """A single orange view among green views produces orange."""
        views = [
            _inference_result(decision="Vert"),
            _inference_result(decision="Orange"),
            _inference_result(decision="Vert"),
        ]
        assert aggregate_piece_predictions(views) == "Orange"

    def test_aggregate_red_presence_overrides_others(self) -> None:
        """A single red view produces red regardless of other views (fail-safe)."""
        views = [
            _inference_result(decision="Vert"),
            _inference_result(decision="Orange"),
            _inference_result(decision="Rouge"),
        ]
        assert aggregate_piece_predictions(views) == "Rouge"

    def test_aggregate_case_insensitive_decision(self) -> None:
        """Aggregation handles different case variations of decision."""
        views = [
            _inference_result(decision="Vert"),
            _inference_result(decision="vert"),  # Lowercase
            _inference_result(decision="VERT"),  # Uppercase
        ]
        result = aggregate_piece_predictions(views)
        assert result in {"Vert", "vert"}
