from __future__ import annotations

import pytest
from fastapi import HTTPException
from prometheus_client.parser import text_string_to_metric_families

from iqa.api.main import OBSERVABILITY, metrics, record_drift_metric_event
from iqa.api.schemas import DriftEventRequest


def _reset_drift_state() -> None:
    OBSERVABILITY.reset()


def test_drift_internal_event_requires_service_token(monkeypatch) -> None:
    _reset_drift_state()
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")

    with pytest.raises(HTTPException) as exc_info:
        record_drift_metric_event(
            DriftEventRequest(
                event_type="window_evaluated",
                scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
            ),
            x_iqa_service_token=None,
        )

    assert exc_info.value.status_code == 401


def test_drift_internal_event_rejects_sensitive_fields(monkeypatch) -> None:
    _reset_drift_state()
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")

    with pytest.raises(HTTPException) as exc_info:
        record_drift_metric_event(
            DriftEventRequest(
                event_type="window_evaluated",
                scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
                metrics={"image_path": 1.0},
            ),
            x_iqa_service_token="service-secret",
        )

    assert exc_info.value.status_code == 422


def test_api_metrics_exposes_drift_metrics(monkeypatch) -> None:
    _reset_drift_state()
    monkeypatch.setenv("IQA_SERVICE_TOKEN", "service-secret")

    response = record_drift_metric_event(
        DriftEventRequest(
            event_type="window_evaluated",
            scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
            status="confirmed",
            source_domain="piece_a_p4",
            window_index=7,
            first_confirmed_window_index=6,
            window_events=60,
            trigger_lifecycle=True,
            active_models={
                "classification": {
                    "version": "rd_feature_ae_gated_natural_cycle_002",
                    "registry_model_name": "feature_ae_classifier__production_replay_natural_piece_b_full",
                    "registered_model_version": "5",
                    "registry_stage": "test",
                    "runtime_contract_status": "loaded",
                },
                "localization": {
                    "version": "rd_feature_ae_gated_natural_cycle_005",
                    "registry_model_name": "feature_ae_localization__production_replay_natural_piece_b_full",
                    "registered_model_version": "11",
                    "registry_stage": "test",
                    "runtime_contract_status": "loaded",
                },
            },
            metrics={
                "drift_score": 0.82,
                "degradation_score": 0.91,
                "domain_score": 1.0,
                "alert_rate": 0.55,
                "red_rate": 0.12,
                "unexpected_red_rate": 0.44,
                "roi_fail_rate": 0.03,
                "oracle_fn_rate": 0.06,
                "domain_ratio": 0.74,
            },
        ),
        x_iqa_service_token="service-secret",
    )

    assert response["accepted"] is True
    body = metrics()
    assert "iqa_drift_score" in body
    assert "iqa_drift_degradation_score" in body
    assert "iqa_drift_domain_score" in body
    assert "iqa_drift_unexpected_red_rate" in body
    assert "iqa_drift_status" in body
    assert 'status="confirmed"' in body
    assert "iqa_drift_window_events" in body
    assert "iqa_drift_window_index" in body
    assert "iqa_drift_first_confirmed_window" in body
    assert "iqa_drift_domain_ratio" in body
    assert "iqa_drift_trigger_lifecycle" in body
    assert "iqa_drift_active_model_info" in body
    assert 'role="classification"' in body
    assert 'runtime_contract_status="loaded"' in body
    assert "aaaaaaaa" not in body
    assert list(text_string_to_metric_families(body))
