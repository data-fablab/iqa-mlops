"""Contracts shared by the API gateway and inference service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from iqa.ingestion.schemas import PieceEvent as PieceEvent  # re-exported


Decision = Literal["Vert", "Orange", "Rouge"]


@dataclass(frozen=True)
class InferenceRequest:
    piece_event_id: str
    scenario_id: str
    image_uri: str


@dataclass(frozen=True)
class InferenceResult:
    piece_event_id: str
    scenario_id: str
    score: float
    decision: Decision
    heatmap_uri: str | None
    roi_status: str | None
    roi_model_version: str
    feature_ae_version: str

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


def placeholder_inference(request: InferenceRequest) -> InferenceResult:
    return InferenceResult(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        score=0.0,
        decision="Vert",
        heatmap_uri=None,
        roi_status=None,
        roi_model_version="roi_segmenter_v001_fixed",
        feature_ae_version="rd_feature_ae_gated_v001_bootstrap",
    )


__all__ = ["Decision", "InferenceRequest", "InferenceResult", "PieceEvent", "placeholder_inference"]
