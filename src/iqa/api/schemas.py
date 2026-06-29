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
    raw_dataset_id: str | None = None
    manifest_id: str | None = None
    dataset_version: str | None = None
    replay_id: str | None = None
    validation_id: str | None = None
    scenario_version: str | None = None
    is_defective_gt: bool | None = None
    views: list[PieceView] = Field(default_factory=list)


class PredictRequest(IQABaseModel):
    piece_event_id: str
    scenario_id: str
    image_uri: str = Field(..., description="S3, DVC or local URI for the primary image.")
    heatmap_uri: str | None = Field(default=None, description="Optional Feature-AE heatmap URI for review display.")
    sha256: str | None = None
    lot_id: str | None = None
    source_class: str | None = None
    dataset_version: str | None = None

    @field_validator("piece_event_id", "scenario_id", "image_uri")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class PieceEventPredictRequest(IQABaseModel):
    scenario_id: str
    image_uri: str = Field(..., description="S3, DVC or local URI for the primary image.")
    heatmap_uri: str | None = Field(default=None, description="Optional Feature-AE heatmap URI for review display.")
    sha256: str | None = None
    lot_id: str | None = None
    source_class: str | None = None
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
    source_class: str | None = None
    raw_dataset_id: str | None = None
    manifest_id: str | None = None
    dataset_version: str | None = None
    replay_id: str | None = None
    validation_id: str | None = None
    scenario_version: str | None = None
    model_version: str | None = None
    roi_model_version: str | None = None
    roi_status: str | None = None
    piece_score: float | None = None
    piece_status: PieceStatus | None = None
    decision_reasons: list[str] = Field(default_factory=list)
    heatmap_uri: str | None = None
    sha256: str | None = None
    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackRequest(IQABaseModel):
    prediction_id: str
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
    train_block_reason: str | None = None
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



class ApiErrorResponse(IQABaseModel):
    error_code: str
    message: str
    status_code: int
    reason: str | None = None
    incident_type: IncidentType | None = None
    audit_logged: bool = False
    reload_event_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ModelVersion(IQABaseModel):
    model_version: str
    model_stage: ModelStage
    registered_model_name: str
    scenario_id: str
    raw_dataset_id: str | None = None
    manifest_id: str | None = None
    dataset_version: str | None = None
    replay_id: str | None = None
    validation_id: str | None = None
    scenario_version: str | None = None
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
    raw_dataset_id: str | None = None
    manifest_id: str | None = None
    dataset_version: str | None = None
    replay_id: str | None = None
    validation_id: str | None = None
    scenario_version: str | None = None


class ReplayRunRequest(IQABaseModel):
    scenario_id: str

    @field_validator("scenario_id")
    @classmethod
    def scenario_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class ReplayRunResponse(IQABaseModel):
    replay_run_id: str
    scenario_id: str
    cursor: int = 0
    total_events: int
    lot_count: int = 0
    lot_ids: list[str] = Field(default_factory=list)
    source_classes: list[str] = Field(default_factory=list)
    finished: bool = False
    created_at: str
    updated_at: str | None = None


class ReplayNextResponse(ReplayRunResponse):
    event: dict[str, Any] | None = None


class ReloadModelRequest(IQABaseModel):
    scenario_id: str
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


class LifecycleEventRequest(IQABaseModel):
    event_type: str
    scenario_id: str
    lifecycle_run_id: str
    cycle_id: str | None = None
    epoch: int | None = None
    candidate_version: str | None = None
    candidate_init_policy: str | None = None
    candidate_initial_model_version: str | None = None
    active_classification_model_version: str | None = None
    active_localization_model_version: str | None = None
    active_classification_registered_model_name: str | None = None
    active_classification_registered_model_version: str | None = None
    active_localization_registered_model_name: str | None = None
    active_localization_registered_model_version: str | None = None
    candidate_initial_checkpoint_sha256: str | None = None
    localization_checkpoint_sha256: str | None = None
    classification_checkpoint_sha256: str | None = None
    localization_promotion_status: str | None = None
    classification_promotion_status: str | None = None
    localization_gate_reason: str | None = None
    classification_gate_reason: str | None = None
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)

    @field_validator("event_type", "scenario_id", "lifecycle_run_id")
    @classmethod
    def lifecycle_field_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class DriftEventRequest(IQABaseModel):
    event_type: str
    scenario_id: str
    status: str = "clear"
    source_domain: str = "piece_a_p4"
    lifecycle_run_id: str | None = None
    cycle_id: str | None = None
    window_index: int | None = None
    first_confirmed_window_index: int | None = None
    window_events: int | None = None
    trigger_lifecycle: bool | None = None
    active_models: dict[str, dict[str, str | int | bool | None]] = Field(default_factory=dict)
    metrics: dict[str, float | int | bool] = Field(default_factory=dict)

    @field_validator("event_type", "scenario_id", "status", "source_domain")
    @classmethod
    def drift_field_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("field must not be empty")
        return value


class AuditTrailPredictionContext(IQABaseModel):
    prediction_id: str
    piece_event_id: str
    scenario_id: str
    lot_id: str | None = None
    source_class: str | None = None
    sha256: str | None = None
    raw_dataset_id: str | None = None
    manifest_id: str | None = None
    dataset_version: str | None = None
    replay_id: str | None = None
    validation_id: str | None = None
    scenario_version: str | None = None
    model_version: str | None = None
    roi_model_version: str | None = None
    decision: str | None = None
    heatmap_uri: str | None = None


class AuditTrailFeedbackContext(IQABaseModel):
    feedback_source: str | None = None
    display_feedback_source: str | None = None
    display_feedback_status: str | None = None
    oracle_verdict: str | None = None
    divergence: str | None = None
    train_eligibility_source: str | None = None
    eligible_for_train: bool | None = None
    train_block_reason: str | None = None
    feedback_closed: bool = False
    conflict_logged: bool = False


class PredictionAuditTrail(IQABaseModel):
    prediction: AuditTrailPredictionContext
    feedback: AuditTrailFeedbackContext


class PredictionHistoryRow(IQABaseModel):
    prediction_id: str
    piece_event_id: str | None = None
    scenario_id: str | None = None
    lot_id: str | None = None
    source_class: str | None = None
    sha256: str | None = None
    raw_dataset_id: str | None = None
    manifest_id: str | None = None
    dataset_version: str | None = None
    replay_id: str | None = None
    validation_id: str | None = None
    scenario_version: str | None = None
    decision: str | None = None
    heatmap_uri: str | None = None
    model_version: str | None = None
    roi_model_version: str | None = None
    created_at: str | None = None
    feedback_closed: bool = False
    display_decision_source: str | None = None
    display_feedback_source: str | None = None
    display_feedback_status: str | None = None
    human_feedback_present: bool = False
    train_eligibility_source: str | None = None
    eligible_for_train: bool | None = None
    train_block_reason: str | None = None
    conflict_logged: bool = False
    oracle_verdict: str | None = None
    divergence: str | None = None
    audit_trail: PredictionAuditTrail


class LotSummaryRow(IQABaseModel):
    lot_id: str
    scenario_id: str
    total: int
    vert: int = 0
    orange: int = 0
    rouge: int = 0
    feedback_closed: int = 0
    divergences: int = 0
    taux_orange: float = 0.0
    taux_rouge: float = 0.0


class AirflowDatasetTaskOutput(IQABaseModel):
    manifest_path: str
    dataset_version: str
    sample_count: int
    filtered_count: int = 0
    roi_status_count: int = 0
    warning: str | None = None


class AirflowTrainTaskOutput(IQABaseModel):
    run_id: str
    checkpoint: str
    run_dir: str


class AirflowEvalTaskOutput(IQABaseModel):
    recall: float = 0.0
    ap: float = 0.0
    orange_rate: float = 0.0
    latency_ms: float = 0.0
    false_negatives: int = 0


class AirflowGatesTaskOutput(IQABaseModel):
    passed: bool
    reason: str | None = None
    gate_results: dict[str, Any] = Field(default_factory=dict)


class AirflowMLflowTaskOutput(IQABaseModel):
    registered_model_name: str
    version: str
    stage: ModelStage = ModelStage.candidate
    run_id: str | None = None
    source_of_truth: str = "mlflow_registry"


class AirflowPromotionTaskOutput(IQABaseModel):
    accepted: bool
    registered_model_name: str | None = None
    version: str | None = None
    stage: ModelStage | None = None
    reason: str | None = None
    source_of_truth: str = "mlflow_registry"


class ModelRegistryRefResponse(IQABaseModel):
    scenario_id: str
    registered_model_name: str
    stage: ModelStage = ModelStage.prod
    source_of_truth: str = "mlflow_registry"


PredictionRequest = PredictRequest


__all__ = [
    "DEFAULT_SCENARIO_ID",
    "ApiErrorResponse",
    "PredictionHistoryRow",
    "PredictionAuditTrail",
    "ReplayNextResponse",
    "ReplayRunRequest",
    "ReplayRunResponse",
    "ModelRegistryRefResponse",
    "LotSummaryRow",
    "AuditTrailPredictionContext",
    "AuditTrailFeedbackContext",
    "AirflowTrainTaskOutput",
    "AirflowPromotionTaskOutput",
    "AirflowMLflowTaskOutput",
    "AirflowGatesTaskOutput",
    "AirflowEvalTaskOutput",
    "AirflowDatasetTaskOutput",
    "DriftEventRequest",
    "FeedbackRequest",
    "FeedbackResponse",
    "FeedbackSource",
    "FeedbackStatus",
    "Incident",
    "IncidentSeverity",
    "IncidentType",
    "LifecycleEventRequest",
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
