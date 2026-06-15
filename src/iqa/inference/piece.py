"""Piece-level prediction aggregation from multi-view image predictions."""

from __future__ import annotations

from typing import Iterable

from iqa.inference.contracts import InferenceResult


def aggregate_piece_predictions(
    image_predictions: Iterable[InferenceResult],
) -> str:
    """Aggregate multi-view image predictions into a piece-level status.

    Uses a fail-safe aggregation rule: the worst (most severe) status wins.
    - If any view is Rouge → Rouge (critical defect)
    - Else if any view is Orange → Orange (potential defect)
    - Else all are Vert → Vert (acceptable)

    Args:
        image_predictions: Iterable of InferenceResult from multiple views of the piece.

    Returns:
        Aggregated status: "Vert", "Orange", or "Rouge".
    """
    predictions = list(image_predictions)

    if not predictions:
        return "Vert"

    statuses = {pred.statut.lower() for pred in predictions}

    if "rouge" in statuses:
        return "Rouge"
    if "orange" in statuses:
        return "Orange"
    return "Vert"


__all__ = ["aggregate_piece_predictions"]
