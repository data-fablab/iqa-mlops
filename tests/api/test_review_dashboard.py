from __future__ import annotations

import pytest

from iqa.api.main import (
    FEEDBACK_STORE,
    PREDICTION_STORE,
    FeedbackRequest,
    PredictRequest,
    feedback,
    list_predictions,
    lots_summary,
    predict,
)


@pytest.fixture(autouse=True)
def _reset_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IQA_SERVICE_TOKEN", raising=False)
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()


def _predict(piece: str, lot: str) -> str:
    response = predict(PredictRequest(piece_event_id=piece, scenario_id=lot, image_uri="s3://b/k.png"))
    return response["prediction"]["prediction_id"]


def test_predictions_route_is_registered() -> None:
    from iqa.api.main import app

    route_paths = {route.path for route in app.routes}
    assert "/predictions" in route_paths
    assert "/lots/summary" in route_paths


def test_list_predictions_flags_faux_negatif_against_oracle() -> None:
    prediction_id = _predict("pe1", "lotA")  # placeholder decision is always "Vert"
    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="pe1",
            scenario_id="lotA",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,  # oracle says defective -> Vert is a faux negatif
        )
    )

    rows = list_predictions()

    assert len(rows) == 1
    assert rows[0]["oracle_verdict"] == "defective"
    assert rows[0]["divergence"] == "faux_negatif"
    assert rows[0]["feedback_closed"] is True


def test_list_predictions_has_no_divergence_without_feedback() -> None:
    _predict("pe2", "lotB")

    rows = list_predictions()

    assert rows[0]["divergence"] is None
    assert rows[0]["oracle_verdict"] is None


def test_lots_summary_aggregates_per_scenario() -> None:
    _predict("pe1", "lotA")
    _predict("pe2", "lotA")
    _predict("pe3", "lotB")

    summary = {row["scenario_id"]: row for row in lots_summary()}

    assert summary["lotA"]["total"] == 2
    assert summary["lotA"]["vert"] == 2
    assert summary["lotB"]["total"] == 1
    assert summary["lotA"]["taux_orange"] == 0.0
