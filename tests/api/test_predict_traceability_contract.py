"""NAT02 traceability contract tests for /predict."""

from __future__ import annotations

import pytest

from iqa.api.main import (
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    PREDICTION_STORE,
    list_predictions,
    lots_summary,
    predict,
    predict_piece_event,
)
from iqa.api.schemas import PieceEventPredictRequest, PredictRequest


@pytest.fixture(autouse=True)
def _reset_stores() -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()


def test_predict_returns_audit_and_store_traceability_fields() -> None:
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat02_001",
            scenario_id="scenario_nat02",
            image_uri="s3://iqa/raw/piece_nat02_001.png",
            sha256="a" * 64,
            lot_id="lot_nat02_001",
            dataset_version="casting_v001",
        )
    )

    prediction = response["prediction"]
    audit = response["audit"]
    prediction_id = prediction["prediction_id"]
    stored = PREDICTION_STORE[prediction_id]

    assert prediction["piece_event_id"] == "piece_nat02_001"
    assert prediction["scenario_id"] == "scenario_nat02"
    assert prediction["lot_id"] == "lot_nat02_001"
    assert prediction["sha256"] == "a" * 64
    assert prediction["dataset_version"] == "casting_v001"
    assert prediction["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert prediction["roi_model_version"] == "roi_segmenter_v001_fixed"

    assert audit["prediction_id"] == prediction_id
    assert audit["piece_event_id"] == "piece_nat02_001"
    assert audit["scenario_id"] == "scenario_nat02"
    assert audit["lot_id"] == "lot_nat02_001"
    assert audit["sha256"] == "a" * 64
    assert audit["dataset_version"] == "casting_v001"
    assert audit["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert audit["roi_model_version"] == "roi_segmenter_v001_fixed"

    assert stored["piece_event_id"] == "piece_nat02_001"
    assert stored["scenario_id"] == "scenario_nat02"
    assert stored["lot_id"] == "lot_nat02_001"
    assert stored["sha256"] == "a" * 64
    assert stored["dataset_version"] == "casting_v001"
    assert stored["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert stored["roi_model_version"] == "roi_segmenter_v001_fixed"


def test_piece_event_predict_keeps_nat02_traceability_fields() -> None:
    response = predict_piece_event(
        "piece_nat02_002",
        PieceEventPredictRequest(
            scenario_id="scenario_nat02",
            image_uri="s3://iqa/raw/piece_nat02_002.png",
            sha256="b" * 64,
            lot_id="lot_nat02_002",
            dataset_version="casting_v002",
        ),
    )

    prediction = response["prediction"]
    audit = response["audit"]
    stored = PREDICTION_STORE[prediction["prediction_id"]]

    assert prediction["piece_event_id"] == "piece_nat02_002"
    assert prediction["lot_id"] == "lot_nat02_002"
    assert prediction["sha256"] == "b" * 64
    assert prediction["dataset_version"] == "casting_v002"

    assert audit["piece_event_id"] == "piece_nat02_002"
    assert audit["lot_id"] == "lot_nat02_002"
    assert audit["sha256"] == "b" * 64
    assert audit["dataset_version"] == "casting_v002"

    assert stored["piece_event_id"] == "piece_nat02_002"
    assert stored["lot_id"] == "lot_nat02_002"
    assert stored["sha256"] == "b" * 64
    assert stored["dataset_version"] == "casting_v002"


def test_prediction_rows_and_lots_summary_expose_nat02_traceability() -> None:
    predict(
        PredictRequest(
            piece_event_id="piece_nat02_003",
            scenario_id="scenario_nat02",
            image_uri="s3://iqa/raw/piece_nat02_003.png",
            sha256="c" * 64,
            lot_id="lot_nat02_group",
            dataset_version="casting_v003",
        )
    )

    rows = list_predictions()
    assert len(rows) == 1
    assert rows[0]["lot_id"] == "lot_nat02_group"
    assert rows[0]["sha256"] == "c" * 64
    assert rows[0]["dataset_version"] == "casting_v003"
    assert rows[0]["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert rows[0]["roi_model_version"] == "roi_segmenter_v001_fixed"

    summary = lots_summary()
    assert len(summary) == 1
    assert summary[0]["lot_id"] == "lot_nat02_group"
    assert summary[0]["scenario_id"] == "scenario_nat02"
    assert summary[0]["total"] == 1
