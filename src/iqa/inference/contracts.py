"""Contracts shared by the API gateway and inference service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from iqa.inference.drift_scoring import decision_for_score, domain_anomaly_score
from iqa.ingestion.schemas import PieceEvent as PieceEvent  # re-exported


Decision = Literal["Vert", "Orange", "Rouge"]


@dataclass(frozen=True)
class InferenceRequest:
    piece_event_id: str
    scenario_id: str
    image_uri: str
    sha256: str | None = None
    lot_id: str | None = None
    source_class: str | None = None
    dataset_version: str | None = None


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
    # PatchCore domain-drift signal (Issue 12), scored alongside the AE. The AE
    # detects defects; this separates product domains (class1 vs class2/class3).
    # ``None`` when the detector is not loaded (e.g. placeholder inference).
    domain_drift_score: float | None = None
    domain_regime: str | None = None

    def to_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


def placeholder_inference(request: InferenceRequest) -> InferenceResult:
    """Deterministic, GPU-free inference for the demo.

    The decision reflects the controlled domain drift: the model is trained on the
    Casting_class1 baseline, so the drift replay's Casting_class2 / Casting_class3
    domain-extension pieces score as anomalies (Orange / Rouge). Any other scenario
    stays in-distribution (Vert). See iqa.inference.drift_scoring.
    """
    score = domain_anomaly_score(request.scenario_id, request.source_class)
    decision = decision_for_score(score)
    return InferenceResult(
        piece_event_id=request.piece_event_id,
        scenario_id=request.scenario_id,
        score=score,
        decision=decision,
        heatmap_uri=None,
        roi_status=None,
        roi_model_version="roi_segmenter_v001_fixed",
        feature_ae_version="rd_feature_ae_gated_v001_bootstrap",
    )


__all__ = ["Decision", "InferenceRequest", "InferenceResult", "PieceEvent", "placeholder_inference"]
