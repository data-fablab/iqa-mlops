"""NAT04 human_sophie display feedback and oracle divergence contracts."""

from __future__ import annotations

import pytest

from metadata_support import get_display_feedback, get_feedback

from iqa.api.main import (
    feedback,
    list_predictions,
    predict,
)
from iqa.api.schemas import FeedbackRequest, FeedbackStatus, PredictRequest


@pytest.fixture(autouse=True)
def _reset_stores() -> None:
    yield


def test_human_sophie_display_feedback_is_visible_before_oracle_gt() -> None:
    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece_nat04_pending",
            scenario_id="scenario_nat04",
            image_uri="s3://iqa/raw/piece_nat04_pending.png",
            sha256="e" * 64,
            lot_id="lot_nat04",
            dataset_version="casting_v004",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    human_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat04_pending",
            scenario_id="scenario_nat04",
            feedback_source="human_sophie",
            feedback_status=FeedbackStatus.conforme_valide,
            comment="Sophie display decision before oracle GT.",
        )
    )

    rows = list_predictions()
    row = rows[0]

    assert human_response["accepted"] is True
    assert human_response["feedback_closed"] is False
    assert human_response["display_decision_source"] == "human_sophie"
    assert human_response["train_eligibility_source"] == "oracle_gt"
    assert human_response["eligible_for_train"] is False
    assert human_response["conflict_logged"] is False

    assert row["prediction_id"] == prediction_id
    assert row["feedback_closed"] is False
    assert row["oracle_verdict"] is None
    assert row["divergence"] is None
    assert row["human_feedback_present"] is True
    assert row["display_feedback_source"] == "human_sophie"
    assert row["display_feedback_status"] == "conforme_valide"
    assert row["display_decision_source"] == "human_sophie"
    assert row["train_eligibility_source"] == "oracle_gt"
    assert row["eligible_for_train"] is False
    assert row["conflict_logged"] is False


def test_human_sophie_then_oracle_gt_divergence_is_journaled_in_predictions() -> None:
    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece_nat04_divergence",
            scenario_id="scenario_nat04",
            image_uri="s3://iqa/raw/piece_nat04_divergence.png",
            sha256="f" * 64,
            lot_id="lot_nat04",
            dataset_version="casting_v004",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat04_divergence",
            scenario_id="scenario_nat04",
            feedback_source="human_sophie",
            feedback_status=FeedbackStatus.conforme_valide,
            comment="Sophie accepted display before oracle GT.",
        )
    )

    gt_response = feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat04_divergence",
            scenario_id="scenario_nat04",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    rows = list_predictions()
    row = rows[0]

    assert gt_response["accepted"] is True
    assert gt_response["feedback_closed"] is True
    assert gt_response["display_decision_source"] == "human_sophie"
    assert gt_response["train_eligibility_source"] == "oracle_gt"
    assert gt_response["eligible_for_train"] is False
    assert gt_response["conflict_logged"] is True

    display_feedback = get_display_feedback(prediction_id)
    oracle_feedback = get_feedback(prediction_id)
    assert display_feedback["feedback_source"] == "human_sophie"
    assert display_feedback["feedback_status"] == "conforme_valide"
    assert oracle_feedback["feedback_source"] == "oracle_gt"
    assert oracle_feedback["conflict_logged"] is True

    assert row["prediction_id"] == prediction_id
    assert row["oracle_verdict"] == "defective"
    assert row["divergence"] == "faux_negatif"
    assert row["human_feedback_present"] is True
    assert row["display_feedback_source"] == "human_sophie"
    assert row["display_feedback_status"] == "conforme_valide"
    assert row["display_decision_source"] == "human_sophie"
    assert row["train_eligibility_source"] == "oracle_gt"
    assert row["eligible_for_train"] is False
    assert row["conflict_logged"] is True
