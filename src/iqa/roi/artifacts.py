"""ROI prediction artifact contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


RoiQualityStatus = Literal["ok", "warning", "fail"]


@dataclass(frozen=True)
class RoiPredictionArtifact:
    piece_event_id: str
    image_id: str
    image_uri: str
    roi_mask_uri: str
    roi_model_version: str
    roi_ratio: float
    roi_quality_status: RoiQualityStatus
    source: str
    scenario_id: str
    dataset_version: str

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


__all__ = ["RoiPredictionArtifact", "RoiQualityStatus"]
