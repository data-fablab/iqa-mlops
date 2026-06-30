"""NAT02 traceability contract tests for /predict."""

from __future__ import annotations

from metadata_support import get_prediction

from iqa.api.main import (
    list_predictions,
    lots_summary,
    predict,
    predict_piece_event,
)
from iqa.api.schemas import (
    ModelStage,
    ModelVersion,
    PieceEvent,
    PieceEventPredictRequest,
    PredictRequest,
    PredictionResponse,
    Scenario,
)


OPTIONAL_METADATA_FIELDS = (
    "raw_dataset_id",
    "manifest_id",
    "replay_id",
    "validation_id",
    "scenario_version",
)


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
    stored = get_prediction(prediction_id)

    assert prediction["piece_event_id"] == "piece_nat02_001"
    assert prediction["scenario_id"] == "scenario_nat02"
    assert prediction["lot_id"] == "lot_nat02_001"
    assert prediction["sha256"] == "a" * 64
    assert prediction["dataset_version"] == "casting_v001"
    assert prediction["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert prediction["roi_model_version"] == "roi_segmenter_v001_fixed"
    assert all(prediction[field] is None for field in OPTIONAL_METADATA_FIELDS)

    assert audit["prediction_id"] == prediction_id
    assert audit["piece_event_id"] == "piece_nat02_001"
    assert audit["scenario_id"] == "scenario_nat02"
    assert audit["lot_id"] == "lot_nat02_001"
    assert audit["sha256"] == "a" * 64
    assert audit["dataset_version"] == "casting_v001"
    assert audit["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert audit["roi_model_version"] == "roi_segmenter_v001_fixed"
    assert all(audit[field] is None for field in OPTIONAL_METADATA_FIELDS)

    assert stored["piece_event_id"] == "piece_nat02_001"
    assert stored["scenario_id"] == "scenario_nat02"
    assert stored["lot_id"] == "lot_nat02_001"
    assert stored["sha256"] == "a" * 64
    assert stored["dataset_version"] == "casting_v001"
    assert stored["model_version"] == "rd_feature_ae_gated_v001_bootstrap"
    assert stored["roi_model_version"] == "roi_segmenter_v001_fixed"
    assert all(stored[field] is None for field in OPTIONAL_METADATA_FIELDS)


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
    stored = get_prediction(prediction["prediction_id"])

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
    assert all(rows[0][field] is None for field in OPTIONAL_METADATA_FIELDS)

    summary = lots_summary()
    assert len(summary) == 1
    assert summary[0]["lot_id"] == "lot_nat02_group"
    assert summary[0]["scenario_id"] == "scenario_nat02"
    assert summary[0]["total"] == 1


def test_traceability_models_accept_phase2_metadata_fields() -> None:
    metadata = {
        "raw_dataset_id": "hss_iad_casting_raw_v1",
        "manifest_id": "casting_flux_replay_plan_natural_v001",
        "dataset_version": "production_replay_natural_v001",
        "replay_id": "production_replay_natural_v001",
        "validation_id": None,
        "scenario_version": "production_replay_natural_v001",
    }

    piece_event = PieceEvent(piece_event_id="sim_event_nat02", scenario_id="production_replay_natural", **metadata)
    prediction = PredictionResponse(piece_event_id="sim_event_nat02", scenario_id="production_replay_natural", **metadata)
    scenario = Scenario(scenario_id="production_replay_natural", scenario_type="production", **metadata)
    model = ModelVersion(
        model_version="rd_feature_ae_gated_v001_bootstrap",
        model_stage=ModelStage.prod,
        registered_model_name="feature_ae__production_replay_natural",
        scenario_id="production_replay_natural",
        **metadata,
    )

    assert piece_event.replay_id == "production_replay_natural_v001"
    assert prediction.manifest_id == "casting_flux_replay_plan_natural_v001"
    assert scenario.scenario_version == "production_replay_natural_v001"
    assert model.raw_dataset_id == "hss_iad_casting_raw_v1"
