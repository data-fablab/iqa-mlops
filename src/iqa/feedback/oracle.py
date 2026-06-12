"""Oracle GT feedback used to automate the MVP workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


FeedbackSource = Literal["oracle_gt", "human_sophie_future"]
QualityVerdict = Literal["conforme", "defective"]


@dataclass(frozen=True)
class OracleFeedbackRequest:
    piece_event_id: str
    scenario_id: str
    gt_mask_uri: str | None = None
    gt_mask_has_defect: bool = False


@dataclass(frozen=True)
class FeedbackVerdict:
    piece_event_id: str
    scenario_id: str
    feedback_source: FeedbackSource
    verdict: QualityVerdict
    train_eligible: bool

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def oracle_gt_verdict(request: OracleFeedbackRequest) -> FeedbackVerdict:
    is_defective = bool(request.gt_mask_has_defect)
    return FeedbackVerdict(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        feedback_source="oracle_gt",
        verdict="defective" if is_defective else "conforme",
        train_eligible=not is_defective,
    )


__all__ = ["FeedbackSource", "FeedbackVerdict", "OracleFeedbackRequest", "QualityVerdict", "oracle_gt_verdict"]
