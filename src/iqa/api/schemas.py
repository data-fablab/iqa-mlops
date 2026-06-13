"""Pydantic schemas for the IQA API.

The schemas define the API contracts for prediction, piece events,
feedback, incidents, model versions, replay scenarios and reload governance.

This file is intentionally MVP compatible with the current API implementation.
Some fields required by the security threat model are prepared here, but their
strict enforcement belongs to the next API and feedback tasks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_SCENARIO_ID = "production_replay_natural"


class PieceStatus(StrEnum):
    vert = "Vert"
    orange = "Orange"
    rouge = "Rouge"


class FeedbackSource(StrEnum):
    oracle_gt = "oracle_gt"
    human_sophie = "human_sophie"


class FeedbackStatus(StrEnum):
    conforme_valide = "conforme_valide"
    defaut_confirme = "defaut_confirme"
    faux_negatif = "faux_negatif"
    faux_positif = "faux_positif"
    roi_warning = "roi_warning"
    roi_fail = "roi_fail"


class IncidentSeverity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class IncidentType(StrEnum):
    false_negative = "false_negative"
    roi_warning = "roi_warning"
    roi_fail = "roi_fail"
    feedback_conflict = "feedback_conflict"
    reload_refused = "reload_refused"
    invalid_prediction_request = "invalid_prediction_request"
    unsafe_train_candidate_blocked = "unsafe_train_candidate_blocked"


class ModelStage(StrEnum):
    candidate = "candidate"
    test = "test"
    prod = "prod"
    archived = "archived"


class ReloadStatus(StrEnum):
    accepted = "accepted"
    refused = "refused"


class IQABaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class PieceView(IQABaseModel):
    image_id: str | None = None
    image_uri: str
    sha256: str | None = None
    view_key: str | None = None
    roi_status: str | None = None


class PieceEvent(IQABaseModel):
    piece_event_id: str
    scenario_id: str = DEFAULT_SCENARIO_ID
    lot_id: str | None = None
    source_class: str | None = None
    group_key: str | None = None
    dataset_version: str | None = None
    is_defective_gt: bool | None = None
    views: list[PieceView] = Field(default_factory=list)


class PredictRequest(IQABaseModel):
    piece_event_id: str
    scenario_id: str = DEFAULT_SCENARIO_ID
    image_uri: str = Field(..., description="S3, DVC or local URI for the primary image.")
    sha256: str | None = None
    lot_id: str | None = None
    dataset_version: str | None = None

    @field_validator("piece_event_id", "scenario_id", "image_uri")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class PieceEventPredictRequest(IQABaseModel):
    scenario_id: str = DEFAULT_SCENARIO_ID
    image_uri: str = Field(..., description="S3, DVC or local URI for the primary image.")
    sha256: str | None = None
    lot_id: str | None = None
    dataset_version: str | None = None

    @field_validator("scenario_id", "image_uri")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class PredictionResponse(IQABaseModel):
    prediction_id: str | None = None
    piece_event_id: str
    scenario_id: str
    lot_id: str | None = None
    dataset_version: str | None = None
    model_version: str | None = None
    roi_model_version: str | None = None
    roi_status: str | None = None
    piece_score: float | None = None
    piece_status: PieceStatus | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    sha256: str | None = None
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackRequest(IQABaseModel):
    prediction_id: str | None = None
    piece_event_id: str
    scenario_id: str = DEFAULT_SCENARIO_ID
    feedback_source: FeedbackSource = FeedbackSource.oracle_gt
    feedback_status: FeedbackStatus | None = None
    human_override: bool = False
    gt_mask_uri: str | None = None
    gt_mask_has_defect: bool = False
    comment: str | None = None

    @field_validator("piece_event_id", "scenario_id")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class FeedbackResponse(IQABaseModel):
    accepted: bool
    prediction_id: str | None = None
    piece_event_id: str
    scenario_id: str
    feedback_source: FeedbackSource
    feedback_status: FeedbackStatus | None = None
    display_decision_source: FeedbackSource | None = None
    train_eligibility_source: FeedbackSource = FeedbackSource.oracle_gt
    eligible_for_train: bool | None = None
    conflict_logged: bool = False
    reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Incident(IQABaseModel):
    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    piece_event_id: str | None = None
    prediction_id: str | None = None
    scenario_id: str | None = None
    model_version: str | None = None
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelVersion(IQABaseModel):
    model_version: str
    model_stage: ModelStage
    registered_model_name: str
    scenario_id: str
    dataset_version: str | None = None
    roi_model_version: str | None = None
    artifact_uri: str | None = None
    source_of_truth: str = "mlflow_registry"
    loaded_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class Scenario(IQABaseModel):
    scenario_id: str
    scenario_type: str
    description: str | None = None
    active: bool = True
    registered_model_name: str | None = None
    dataset_version: str | None = None


class ReloadModelRequest(IQABaseModel):
    scenario_id: str = DEFAULT_SCENARIO_ID
    stage: ModelStage = ModelStage.prod
    requested_by: str | None = None
    reason: str | None = None

    @field_validator("scenario_id")
    @classmethod
    def scenario_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class ReloadModelResponse(IQABaseModel):
    accepted: bool
    reload_status: ReloadStatus
    scenario_id: str
    previous_model_version: str | None = None
    new_model_version: str | None = None
    source_of_truth: str = "mlflow_registry"
    reason: str | None = None
    audit_logged: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


PredictionRequest = PredictRequest


__all__ = [
    "DEFAULT_SCENARIO_ID",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackSource",
    "FeedbackStatus",
    "Incident",
    "IncidentSeverity",
    "IncidentType",
    "ModelStage",
    "ModelVersion",
    "PieceEvent",
    "PieceEventPredictRequest",
    "PieceStatus",
    "PieceView",
    "PredictRequest",
    "PredictionRequest",
    "PredictionResponse",
    "ReloadModelRequest",
    "ReloadModelResponse",
    "ReloadStatus",
    "Scenario",
]
