"""NAT10 filtered Prometheus metrics contracts."""

from __future__ import annotations

import pytest

from iqa.api.main import (
    AI_SECURITY_METRICS,
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    PREDICTION_METRICS,
    PREDICTION_STORE,
    feedback,
    list_predictions,
    metrics,
    predict,
)
from iqa.api.schemas import FeedbackRequest, PredictRequest


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    for key in PREDICTION_METRICS:
        PREDICTION_METRICS[key] = 0
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()


def test_nat10_predict_propagates_source_class_to_traceability_fields() -> None:
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat10_trace",
            scenario_id="scenario_nat10",
            image_uri="s3://iqa/raw/piece_nat10_trace.png",
            sha256="1" * 64,
            lot_id="lot_nat10",
            source_class="Casting_class1",
            dataset_version="casting_v010",
        )
    )

    prediction = response["prediction"]
    audit = response["audit"]
    prediction_id = prediction["prediction_id"]
    row = list_predictions()[0]

    assert prediction["source_class"] == "Casting_class1"
    assert audit["source_class"] == "Casting_class1"
    assert PREDICTION_STORE[prediction_id]["source_class"] == "Casting_class1"
    assert row["source_class"] == "Casting_class1"
    assert row["audit_trail"]["prediction"]["source_class"] == "Casting_class1"


def test_nat10_metrics_are_filterable_by_trace_labels() -> None:
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat10_metric",
            scenario_id="scenario_nat10",
            image_uri="s3://iqa/raw/piece_nat10_metric.png",
            sha256="2" * 64,
            lot_id="lot_nat10",
            source_class="Casting_class1",
            dataset_version="casting_v010",
        )
    )
    prediction_id = response["prediction"]["prediction_id"]

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat10_metric",
            scenario_id="scenario_nat10",
            feedback_source="oracle_gt",
            gt_mask_has_defect=False,
        )
    )

    label_base = (
        'scenario_id="scenario_nat10",'
        'lot_id="lot_nat10",'
        'source_class="Casting_class1",'
        'model_version="rd_feature_ae_gated_v001_bootstrap",'
        'dataset_version="casting_v010"'
    )
    body = metrics()

    assert f'iqa_prediction_filtered_total{{{label_base},decision="Vert"}} 1' in body
    assert f'iqa_feedback_closed_filtered_total{{{label_base}}} 1' in body
    assert f'iqa_train_eligible_filtered_total{{{label_base}}} 1' in body


def test_nat10_divergence_metrics_are_filterable_by_trace_labels() -> None:
    response = predict(
        PredictRequest(
            piece_event_id="piece_nat10_divergence",
            scenario_id="scenario_nat10",
            image_uri="s3://iqa/raw/piece_nat10_divergence.png",
            sha256="3" * 64,
            lot_id="lot_nat10_div",
            source_class="Casting_crack",
            dataset_version="casting_v010",
        )
    )
    prediction_id = response["prediction"]["prediction_id"]
    PREDICTION_STORE[prediction_id]["decision"] = "Vert"

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat10_divergence",
            scenario_id="scenario_nat10",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    label_base = (
        'scenario_id="scenario_nat10",'
        'lot_id="lot_nat10_div",'
        'source_class="Casting_crack",'
        'model_version="rd_feature_ae_gated_v001_bootstrap",'
        'dataset_version="casting_v010"'
    )
    body = metrics()

    assert f'iqa_prediction_filtered_total{{{label_base},decision="Vert"}} 1' in body
    assert f'iqa_feedback_closed_filtered_total{{{label_base}}} 1' in body
    assert f'iqa_divergence_filtered_total{{{label_base},divergence="faux_negatif"}} 1' in body
    assert f'iqa_train_eligible_filtered_total{{{label_base}}}' not in body
