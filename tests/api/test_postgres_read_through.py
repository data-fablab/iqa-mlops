from __future__ import annotations

from copy import deepcopy

import pytest

from iqa.api import main as api
from iqa.api.schemas import FeedbackRequest


class RestartRepository:
    def __init__(
        self,
        prediction: dict,
        display_feedback: dict | None = None,
    ) -> None:
        prediction_id = prediction["prediction_id"]
        self.predictions = {prediction_id: deepcopy(prediction)}
        self.feedbacks: dict[str, dict] = {}
        self.display_feedbacks = (
            {prediction_id: deepcopy(display_feedback)}
            if display_feedback is not None
            else {}
        )

    def get_prediction(self, prediction_id: str) -> dict | None:
        record = self.predictions.get(prediction_id)
        return deepcopy(record) if record is not None else None

    def list_predictions(self) -> list[dict]:
        return [deepcopy(record) for record in self.predictions.values()]

    def get_feedback(self, prediction_id: str) -> dict | None:
        record = self.feedbacks.get(prediction_id)
        return deepcopy(record) if record is not None else None

    def get_display_feedback(self, prediction_id: str) -> dict | None:
        record = self.display_feedbacks.get(prediction_id)
        return deepcopy(record) if record is not None else None

    def save_feedback_and_close_prediction(
        self,
        prediction_id: str,
        feedback_record: dict,
        closed_at: str,
    ) -> None:
        self.feedbacks[prediction_id] = deepcopy(feedback_record)
        self.predictions[prediction_id]["feedback_closed"] = True
        self.predictions[prediction_id]["feedback_closed_at"] = closed_at


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch: pytest.MonkeyPatch) -> None:
    api.METADATA_REPOSITORY.reset()
    monkeypatch.setenv("IQA_METADATA_BACKEND", "postgres")
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    yield
    api.METADATA_REPOSITORY.reset()


def prediction_record() -> dict:
    return {
        "prediction_id": "pred_nat14_001",
        "piece_event_id": "piece_nat14_001",
        "scenario_id": "scenario_nat14",
        "lot_id": "lot_nat14",
        "source_class": "Casting_class3",
        "dataset_version": "casting_nat14_v001",
        "decision": "Vert",
        "model_version": "rd_feature_ae_gated_v001_bootstrap",
        "roi_model_version": "roi_segmenter_v001_fixed",
        "feedback_closed": False,
        "feedback_closed_at": None,
        "created_at": "2026-06-26T12:00:00+00:00",
    }


def display_feedback_record() -> dict:
    return {
        "prediction_id": "pred_nat14_001",
        "piece_event_id": "piece_nat14_001",
        "scenario_id": "scenario_nat14",
        "feedback_source": "human_sophie",
        "feedback_status": "a_revoir",
        "display_decision_source": "human_sophie",
        "train_eligibility_source": "oracle_gt",
        "eligible_for_train": False,
        "train_block_reason": "human_sophie_display_only",
        "conflict_logged": False,
    }


def test_oracle_feedback_recovers_postgres_state_after_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = RestartRepository(
        prediction_record(),
        display_feedback_record(),
    )
    monkeypatch.setattr(api, "create_metadata_repository", lambda: repository)

    response = api.feedback(
        FeedbackRequest(
            prediction_id="pred_nat14_001",
            piece_event_id="piece_nat14_001",
            scenario_id="scenario_nat14",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        )
    )

    assert response["accepted"] is True
    assert response["feedback_closed"] is True
    assert response["display_decision_source"] == "human_sophie"
    assert response["conflict_logged"] is True
    assert repository.predictions["pred_nat14_001"]["feedback_closed"] is True
    assert repository.feedbacks["pred_nat14_001"]["feedback_source"] == "oracle_gt"


def test_prediction_history_recovers_postgres_state_after_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = RestartRepository(
        prediction_record(),
        display_feedback_record(),
    )
    monkeypatch.setattr(api, "create_metadata_repository", lambda: repository)

    rows = api.list_predictions()

    assert len(rows) == 1
    assert rows[0]["prediction_id"] == "pred_nat14_001"
    assert rows[0]["human_feedback_present"] is True
    assert rows[0]["display_feedback_source"] == "human_sophie"
    assert rows[0]["display_feedback_status"] == "a_revoir"
