"""Runtime inference pipeline contracts for ROI then Feature-AE."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from iqa.inference.contracts import Decision, InferenceRequest, InferenceResult
from iqa.roi.artifacts import RoiPredictionArtifact


@dataclass(frozen=True)
class InferencePipelineResult:
    request: InferenceRequest
    roi_prediction: RoiPredictionArtifact | None
    result: InferenceResult

    def to_dict(self) -> dict:
        return {
            "request": asdict(self.request),
            "roi_prediction": None if self.roi_prediction is None else self.roi_prediction.to_dict(),
            "result": self.result.to_dict(),
        }


def decision_from_roi_and_score(roi_quality_status: str, score: float, *, orange_threshold: float, red_threshold: float) -> Decision:
    if roi_quality_status == "fail" or score >= red_threshold:
        return "Rouge"
    if roi_quality_status == "warning" or score >= orange_threshold:
        return "Orange"
    return "Vert"


__all__ = ["InferencePipelineResult", "decision_from_roi_and_score"]
