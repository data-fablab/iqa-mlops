"""NAT16 AI security governance contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from iqa.api.main import (
    ADMIN_RELOAD_LOG,
    AI_SECURITY_METRICS,
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    INCIDENT_STORE,
    PREDICTION_METRICS,
    PREDICTION_STORE,
    feedback,
    list_incidents,
    list_predictions,
    metrics,
    predict,
)
from iqa.api.schemas import FeedbackRequest, FeedbackStatus, PredictRequest
from iqa.datasets import filter_candidate_samples, validate_good_only_samples


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    INCIDENT_STORE.clear()
    ADMIN_RELOAD_LOG.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    for key in PREDICTION_METRICS:
        PREDICTION_METRICS[key] = 0
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("IQA_ADMIN_TOKEN", raising=False)
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    INCIDENT_STORE.clear()
    ADMIN_RELOAD_LOG.clear()


def _sample(
    image_id: str,
    *,
    label: str = "good",
    is_defective: bool = False,
    split_set: str = "train",
    relative_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        image_id=image_id,
        relative_path=relative_path or f"{image_id}.png",
        event_id=f"event_{image_id}",
        source_class="Casting_class1",
        split_set=split_set,
        label=label,
        is_defective=is_defective,
        scenario_id="scenario_nat16",
        dataset_version="casting_v016",
        gt_mask_path="",
    )


def _create_prediction(
    *,
    piece_event_id: str = "piece_nat16",
    scenario_id: str = "scenario_nat16",
) -> str:
    response = predict(
        PredictRequest(
            piece_event_id=piece_event_id,
            scenario_id=scenario_id,
            image_uri=f"s3://iqa/raw/{piece_event_id}.png",
            sha256="7" * 64,
            lot_id="lot_nat16",
            source_class="Casting_class1",
            dataset_version="casting_v016",
        )
    )
    return response["prediction"]["prediction_id"]


def test_nat16_unknown_prediction_is_blocked_before_feedback_state_mutation() -> None:
    with pytest.raises(HTTPException) as exc_info:
        feedback(
            FeedbackRequest(
                prediction_id="pred_nat16_unknown",
                piece_event_id="piece_nat16_unknown",
                scenario_id="scenario_nat16",
                feedback_source="oracle_gt",
                gt_mask_has_defect=True,
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail["error_code"] == "prediction_not_found"
    assert exc_info.value.detail["incident_type"] == "invalid_prediction_request"
    assert FEEDBACK_STORE == {}
    assert DISPLAY_FEEDBACK_STORE == {}
    assert list_incidents() == []

    body = metrics()
    assert "iqa_invalid_feedback_total 1" in body
    assert "iqa_ai_security_incident_total 1" in body


def test_nat16_human_sophie_cannot_make_feedback_train_eligible_before_oracle_gt() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat16_human")

    human_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat16_human",
            scenario_id="scenario_nat16",
            feedback_source="human_sophie",
            feedback_status=FeedbackStatus.conforme_valide,
            comment="Display validation only.",
        )
    )

    assert human_response["accepted"] is True
    assert human_response["feedback_closed"] is False
    assert human_response["display_decision_source"] == "human_sophie"
    assert human_response["train_eligibility_source"] == "oracle_gt"
    assert human_response["eligible_for_train"] is False

    row = list_predictions()[0]
    assert row["human_feedback_present"] is True
    assert row["feedback_closed"] is False
    assert row["eligible_for_train"] is False
    assert row["train_block_reason"] == "human_sophie_display_only"
    assert prediction_id not in FEEDBACK_STORE
    assert DISPLAY_FEEDBACK_STORE[prediction_id]["feedback_source"] == "human_sophie"
    assert "iqa_unsafe_train_blocked_total 1" in metrics()


def test_nat16_oracle_gt_false_negative_creates_security_incident_and_blocks_training() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat16_false_negative")
    PREDICTION_STORE[prediction_id]["decision"] = "Vert"

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat16_false_negative",
            scenario_id="scenario_nat16",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    assert response["accepted"] is True
    assert response["feedback_closed"] is True
    assert response["eligible_for_train"] is False
    assert response["train_block_reason"] == "oracle_gt_defective"

    rows = list_predictions()
    assert rows[0]["divergence"] == "faux_negatif"
    assert rows[0]["eligible_for_train"] is False

    incidents = list_incidents(incident_type="false_negative")
    assert len(incidents) == 1
    assert incidents[0]["severity"] == "high"
    assert incidents[0]["prediction_id"] == prediction_id
    assert incidents[0]["metadata"]["divergence"] == "faux_negatif"
    assert incidents[0]["metadata"]["oracle_verdict"] == "defective"
    assert "iqa_unsafe_train_blocked_total 1" in metrics()


def test_nat16_roi_warning_blocks_train_but_does_not_create_roi_fail_incident() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat16_roi_warning")

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat16_roi_warning",
            scenario_id="scenario_nat16",
            feedback_source="oracle_gt",
            feedback_status=FeedbackStatus.roi_warning,
            gt_mask_has_defect=False,
        )
    )

    assert response["accepted"] is True
    assert response["feedback_closed"] is True
    assert response["eligible_for_train"] is False
    assert response["train_block_reason"] == "roi_warning"
    assert FEEDBACK_STORE[prediction_id]["train_block_reason"] == "roi_warning"
    assert list_incidents(incident_type="roi_fail") == []
    assert "iqa_unsafe_train_blocked_total 1" in metrics()


def test_nat16_candidate_dataset_filter_blocks_defective_roi_fail_and_validation_samples() -> None:
    samples = [
        _sample("img_good_ok", label="good", is_defective=False, split_set="train"),
        _sample("img_normal_ok", label="normal", is_defective=False, split_set="train"),
        _sample("img_defective", label="good", is_defective=True, split_set="train"),
        _sample("img_bad_label", label="anomaly", is_defective=False, split_set="train"),
        _sample("img_validation", label="good", is_defective=False, split_set="validation_set_v001"),
        _sample("img_roi_fail", label="good", is_defective=False, split_set="train"),
    ]
    roi_status = {
        "img_good_ok": "ok",
        "img_normal_ok": "ok",
        "img_defective": "ok",
        "img_bad_label": "ok",
        "img_validation": "ok",
        "img_roi_fail": "fail",
    }

    filtered = filter_candidate_samples(samples, roi_status=roi_status)
    filtered_ids = {sample.image_id for sample in filtered}

    assert filtered_ids == {"img_good_ok", "img_normal_ok"}
    assert "img_defective" not in filtered_ids
    assert "img_bad_label" not in filtered_ids
    assert "img_validation" not in filtered_ids
    assert "img_roi_fail" not in filtered_ids


def test_nat16_good_only_training_validation_rejects_poisoned_training_samples() -> None:
    with pytest.raises(ValueError, match="Feature-AE training accepts only normal"):
        validate_good_only_samples(
            [
                _sample("img_good", label="good", is_defective=False, split_set="train"),
                _sample("img_defective_poison", label="good", is_defective=True, split_set="train"),
            ]
        )

    with pytest.raises(ValueError, match="Feature-AE training accepts only normal"):
        validate_good_only_samples(
            [
                _sample("img_validation_poison", label="good", is_defective=False, split_set="validation_set_v001"),
            ]
        )
