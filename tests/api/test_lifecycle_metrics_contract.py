from __future__ import annotations

import pytest
from fastapi import HTTPException
from prometheus_client.parser import text_string_to_metric_families

from iqa.api.main import LIFECYCLE_STATE, metrics, record_lifecycle_metric_event
from iqa.api.schemas import LifecycleEventRequest


def _reset_lifecycle_state() -> None:
    LIFECYCLE_STATE["current"].clear()
    LIFECYCLE_STATE["epoch_metrics"].clear()
    LIFECYCLE_STATE["epoch_updated_at"] = 0.0
    LIFECYCLE_STATE["gate_metrics"].clear()
    LIFECYCLE_STATE["gate_values"].clear()
    LIFECYCLE_STATE["gate_deltas"].clear()
    LIFECYCLE_STATE["active_models"].clear()
    LIFECYCLE_STATE["final_models"].clear()
    LIFECYCLE_STATE["summary_metrics"].clear()
    LIFECYCLE_STATE["promotion_decisions"].clear()
    LIFECYCLE_STATE["promotion_seen"].clear()
    LIFECYCLE_STATE["promotion_counters"].clear()


def test_lifecycle_internal_event_requires_service_token(monkeypatch) -> None:
    _reset_lifecycle_state()
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")

    with pytest.raises(HTTPException) as exc_info:
        record_lifecycle_metric_event(
            LifecycleEventRequest(
                event_type="run_started",
                scenario_id="production_replay_natural_piece_b_full",
                lifecycle_run_id="replay_lifecycle_001",
            ),
            x_iqa_service_token=None,
        )

    assert exc_info.value.status_code == 401


def test_lifecycle_internal_event_rejects_sensitive_fields(monkeypatch) -> None:
    _reset_lifecycle_state()
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")

    with pytest.raises(HTTPException) as exc_info:
        record_lifecycle_metric_event(
            LifecycleEventRequest(
                event_type="epoch_completed",
                scenario_id="production_replay_natural_piece_b_full",
                lifecycle_run_id="replay_lifecycle_001",
                cycle_id="cycle_001",
                candidate_version="rd_feature_ae_gated_natural_cycle_001",
                metrics={"image_path": 1.0},
            ),
            x_iqa_service_token="service-secret",
        )

    assert exc_info.value.status_code == 422


def test_api_metrics_exposes_lifecycle_metrics(monkeypatch) -> None:
    _reset_lifecycle_state()
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")

    epoch_response = record_lifecycle_metric_event(
        LifecycleEventRequest(
            event_type="epoch_completed",
            scenario_id="production_replay_natural_piece_b_full",
            lifecycle_run_id="replay_lifecycle_001",
            cycle_id="cycle_002",
            epoch=4,
            candidate_version="rd_feature_ae_gated_natural_cycle_002",
            candidate_init_policy="active",
            candidate_initial_model_version="rd_feature_ae_gated_natural_cycle_001",
            active_classification_model_version="rd_feature_ae_gated_natural_cycle_001",
            active_localization_model_version="rd_feature_ae_gated_natural_cycle_005",
            active_classification_registered_model_name="feature_ae_classifier__production_replay_natural_piece_b_full",
            active_classification_registered_model_version="2",
            active_localization_registered_model_name="feature_ae_localization__production_replay_natural_piece_b_full",
            active_localization_registered_model_version="5",
            candidate_initial_checkpoint_sha256="a" * 64,
            metrics={
                "pixel_aupimo_1e-5_1e-3": 0.12,
                "pixel_ap": 0.34,
                "image_ap": 0.56,
                "false_negatives": 2,
            },
        ),
        x_iqa_service_token="service-secret",
    )
    gate_response = record_lifecycle_metric_event(
        LifecycleEventRequest(
            event_type="promotion_decision",
            scenario_id="production_replay_natural_piece_b_full",
            lifecycle_run_id="replay_lifecycle_001",
            cycle_id="cycle_002",
            candidate_version="rd_feature_ae_gated_natural_cycle_002",
            candidate_init_policy="active",
            localization_promotion_status="promoted",
            classification_promotion_status="rejected_no_classification_improvement",
            metrics={
                "localization_metric_delta": 0.05,
                "classification_metric_delta": -1,
                "classification_fn_delta": 1,
                "gate_localization_active_pixel_aupimo": 0.10,
                "gate_localization_candidate_pixel_aupimo": 0.15,
                "gate_delta_localization_pixel_aupimo": 0.05,
                "gate_classification_active_false_negatives": 1,
                "gate_classification_candidate_false_negatives": 2,
                "gate_delta_classification_false_negatives": 1,
                "gate_classification_active_image_ap": 0.70,
                "gate_classification_candidate_image_ap": 0.80,
                "gate_delta_classification_image_ap": 0.10,
            },
        ),
        x_iqa_service_token="service-secret",
    )
    completed_response = record_lifecycle_metric_event(
        LifecycleEventRequest(
            event_type="run_completed",
            scenario_id="production_replay_natural_piece_b_full",
            lifecycle_run_id="replay_lifecycle_001",
            cycle_id="cycle_002",
            candidate_version="rd_feature_ae_gated_natural_cycle_002",
            active_classification_model_version="rd_feature_ae_gated_natural_cycle_002",
            active_classification_registered_model_name="feature_ae_classifier__production_replay_natural_piece_b_full",
            active_classification_registered_model_version="3",
            active_localization_model_version="rd_feature_ae_gated_natural_cycle_005",
            active_localization_registered_model_name="feature_ae_localization__production_replay_natural_piece_b_full",
            active_localization_registered_model_version="6",
            metrics={
                "events_processed": 372,
                "cycles_completed": 5,
            },
        ),
        x_iqa_service_token="service-secret",
    )

    assert epoch_response["accepted"] is True
    assert gate_response["accepted"] is True
    assert completed_response["accepted"] is True
    body = metrics()
    assert "iqa_lifecycle_cycle_current" in body
    assert "iqa_lifecycle_epoch_current" in body
    assert "iqa_lifecycle_epoch_pixel_aupimo" in body
    assert "iqa_lifecycle_epoch_metric" in body
    assert 'metric="false_negatives"' in body
    assert 'iqa_lifecycle_gate_metric_delta{' in body
    assert 'iqa_lifecycle_gate_value{' in body
    assert 'model="active"' in body
    assert 'metric="pixel_aupimo"' in body
    assert 'iqa_lifecycle_gate_delta{' in body
    assert 'role="localization"' in body
    assert 'role="classification",version="rd_feature_ae_gated_natural_cycle_002"' in body
    assert 'role="localization",version="rd_feature_ae_gated_natural_cycle_005"' in body
    assert "iqa_lifecycle_promotion_total" in body
    assert "iqa_lifecycle_promotion_decision_info" in body
    assert "iqa_lifecycle_final_model_info" in body
    assert "iqa_lifecycle_run_events_processed" in body
    assert "iqa_lifecycle_run_cycles_completed" in body
    assert 'status="promoted"' in body
    assert "aaaaaaaa" not in body
    assert list(text_string_to_metric_families(body))
