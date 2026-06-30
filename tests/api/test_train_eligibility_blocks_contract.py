"""NAT07 train eligibility blocking contracts."""

from __future__ import annotations

import pytest

from metadata_support import get_feedback, set_prediction_field

from iqa.api.main import (
    AI_SECURITY_METRICS,
    feedback,
    list_predictions,
    metrics,
    predict,
)
from iqa.api.schemas import FeedbackRequest, FeedbackStatus, PredictRequest


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    yield


def _create_prediction(piece_event_id: str = "piece_nat07", scenario_id: str = "scenario_nat07") -> str:
    response = predict(
        PredictRequest(
            piece_event_id=piece_event_id,
            scenario_id=scenario_id,
            image_uri="s3://iqa/raw/piece_nat07.png",
            sha256="7" * 64,
            lot_id="lot_nat07",
            dataset_version="casting_v007",
        )
    )
    return response["prediction"]["prediction_id"]


def test_nat07_allows_conforming_oracle_feedback_for_train() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat07_conforme")

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat07_conforme",
            scenario_id="scenario_nat07",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        )
    )

    assert response["accepted"] is True
    assert response["train_eligibility_source"] == "oracle_gt"
    assert response["eligible_for_train"] is True
    assert response["train_block_reason"] is None
    assert get_feedback(prediction_id)["eligible_for_train"] is True
    assert get_feedback(prediction_id)["train_block_reason"] is None


def test_nat07_blocks_defective_oracle_feedback_and_faux_negatif_row() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat07_faux_negatif")
    set_prediction_field(prediction_id, "decision", "Vert")

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat07_faux_negatif",
            scenario_id="scenario_nat07",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    row = list_predictions()[0]

    assert response["accepted"] is True
    assert response["eligible_for_train"] is False
    assert response["train_block_reason"] == "oracle_gt_defective"
    assert row["divergence"] == "faux_negatif"
    assert row["eligible_for_train"] is False
    assert row["train_block_reason"] == "oracle_gt_defective"
    assert "iqa_unsafe_train_blocked_total 1" in metrics()


@pytest.mark.parametrize(
    ("feedback_status", "expected_reason"),
    [
        (FeedbackStatus.defaut_confirme, "feedback_status_defaut_confirme"),
        (FeedbackStatus.faux_negatif, "feedback_status_faux_negatif"),
        (FeedbackStatus.roi_warning, "roi_warning"),
        (FeedbackStatus.roi_fail, "roi_fail"),
    ],
)
def test_nat07_blocks_unsafe_feedback_statuses_from_train(
    feedback_status: FeedbackStatus,
    expected_reason: str,
) -> None:
    prediction_id = _create_prediction(piece_event_id=f"piece_nat07_{feedback_status.value}")

    response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id=f"piece_nat07_{feedback_status.value}",
            scenario_id="scenario_nat07",
            feedback_source="oracle_gt",
            feedback_status=feedback_status,
            gt_mask_has_defect=False,
        )
    )

    assert response["accepted"] is True
    assert response["train_eligibility_source"] == "oracle_gt"
    assert response["eligible_for_train"] is False
    assert response["train_block_reason"] == expected_reason
    assert get_feedback(prediction_id)["train_block_reason"] == expected_reason


def test_nat07_exposes_train_block_reason_in_predictions() -> None:
    prediction_id = _create_prediction(piece_event_id="piece_nat07_roi_fail")

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat07_roi_fail",
            scenario_id="scenario_nat07",
            feedback_source="oracle_gt",
            feedback_status=FeedbackStatus.roi_fail,
            gt_mask_has_defect=False,
        )
    )

    row = list_predictions()[0]

    assert row["prediction_id"] == prediction_id
    assert row["eligible_for_train"] is False
    assert row["train_block_reason"] == "roi_fail"
