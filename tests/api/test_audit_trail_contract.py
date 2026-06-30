"""NAT09 audit trail contracts."""

from __future__ import annotations

import pytest

from metadata_support import get_display_feedback, get_feedback, set_prediction_field

from iqa.api.main import (
    AI_SECURITY_METRICS,
    feedback,
    list_predictions,
    predict,
)
from iqa.api.schemas import FeedbackRequest, FeedbackStatus, PredictRequest


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    yield


def test_nat09_audit_trail_links_sha256_piece_scenario_lot_model_and_feedback() -> None:
    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece_nat09_001",
            scenario_id="scenario_nat09",
            image_uri="s3://iqa/raw/piece_nat09_001.png",
            sha256="9" * 64,
            lot_id="lot_nat09",
            dataset_version="casting_v009",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]
    set_prediction_field(prediction_id, "decision", "Vert")

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat09_001",
            scenario_id="scenario_nat09",
            feedback_source="human_sophie",
            feedback_status=FeedbackStatus.conforme_valide,
            comment="Display feedback before oracle GT.",
        )
    )

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat09_001",
            scenario_id="scenario_nat09",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    row = list_predictions()[0]
    audit_trail = row["audit_trail"]

    assert row["prediction_id"] == prediction_id
    assert row["piece_event_id"] == "piece_nat09_001"
    assert row["scenario_id"] == "scenario_nat09"
    assert row["lot_id"] == "lot_nat09"
    assert row["sha256"] == "9" * 64
    assert row["dataset_version"] == "casting_v009"
    assert row["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert row["roi_model_version"] == "roi_segmenter_v001_fixed"

    assert audit_trail["prediction"]["prediction_id"] == prediction_id
    assert audit_trail["prediction"]["piece_event_id"] == "piece_nat09_001"
    assert audit_trail["prediction"]["scenario_id"] == "scenario_nat09"
    assert audit_trail["prediction"]["lot_id"] == "lot_nat09"
    assert audit_trail["prediction"]["sha256"] == "9" * 64
    assert audit_trail["prediction"]["dataset_version"] == "casting_v009"
    assert audit_trail["prediction"]["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert audit_trail["prediction"]["roi_model_version"] == "roi_segmenter_v001_fixed"

    assert audit_trail["feedback"]["feedback_source"] == "oracle_gt"
    assert audit_trail["feedback"]["display_feedback_source"] == "human_sophie"
    assert audit_trail["feedback"]["display_feedback_status"] == "conforme_valide"
    assert audit_trail["feedback"]["oracle_verdict"] == "defective"
    assert audit_trail["feedback"]["divergence"] == "faux_negatif"
    assert audit_trail["feedback"]["train_eligibility_source"] == "oracle_gt"
    assert audit_trail["feedback"]["eligible_for_train"] is False
    assert audit_trail["feedback"]["train_block_reason"] == "oracle_gt_defective"
    assert audit_trail["feedback"]["feedback_closed"] is True
    assert audit_trail["feedback"]["conflict_logged"] is True

    display_feedback = get_display_feedback(prediction_id)
    oracle_feedback = get_feedback(prediction_id)
    assert display_feedback["sha256"] == "9" * 64
    assert display_feedback["lot_id"] == "lot_nat09"
    assert display_feedback["dataset_version"] == "casting_v009"
    assert oracle_feedback["sha256"] == "9" * 64
    assert oracle_feedback["lot_id"] == "lot_nat09"
    assert oracle_feedback["dataset_version"] == "casting_v009"
    assert oracle_feedback["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert oracle_feedback["roi_model_version"] == "roi_segmenter_v001_fixed"
