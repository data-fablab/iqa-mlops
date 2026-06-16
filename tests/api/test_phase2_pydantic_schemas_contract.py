"""NAT12 Phase 2 Pydantic schema contracts."""

from __future__ import annotations

import pytest

from iqa.api.main import (
    AI_SECURITY_METRICS,
    DISPLAY_FEEDBACK_STORE,
    FEEDBACK_STORE,
    PREDICTION_STORE,
    feedback,
    list_predictions,
    lots_summary,
    predict,
)
from iqa.api.schemas import (
    AirflowDatasetTaskOutput,
    AirflowEvalTaskOutput,
    AirflowGatesTaskOutput,
    AirflowMLflowTaskOutput,
    AirflowPromotionTaskOutput,
    AirflowTrainTaskOutput,
    FeedbackRequest,
    FeedbackStatus,
    LotSummaryRow,
    ModelRegistryRefResponse,
    ModelStage,
    PredictRequest,
    PredictionHistoryRow,
)
from iqa.registry import ModelRegistryRef


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()
    for key in AI_SECURITY_METRICS:
        AI_SECURITY_METRICS[key] = 0
    yield
    PREDICTION_STORE.clear()
    FEEDBACK_STORE.clear()
    DISPLAY_FEEDBACK_STORE.clear()


def test_nat12_streamlit_prediction_and_lot_rows_match_pydantic_contracts() -> None:
    prediction_response = predict(
        PredictRequest(
            piece_event_id="piece_nat12_streamlit",
            scenario_id="scenario_nat12",
            image_uri="s3://iqa/raw/piece_nat12_streamlit.png",
            sha256="1" * 64,
            lot_id="lot_nat12",
            source_class="Casting_class1",
            dataset_version="casting_v012",
        )
    )
    prediction_id = prediction_response["prediction"]["prediction_id"]
    PREDICTION_STORE[prediction_id]["decision"] = "Vert"

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat12_streamlit",
            scenario_id="scenario_nat12",
            feedback_source="human_sophie",
            feedback_status=FeedbackStatus.conforme_valide,
        )
    )

    feedback(
        FeedbackRequest(
            prediction_id=prediction_id,
            piece_event_id="piece_nat12_streamlit",
            scenario_id="scenario_nat12",
            feedback_source="oracle_gt",
            gt_mask_has_defect=True,
        )
    )

    prediction_row = PredictionHistoryRow.model_validate(list_predictions()[0])
    lot_row = LotSummaryRow.model_validate(lots_summary()[0])

    assert prediction_row.audit_trail.prediction.sha256 == "1" * 64
    assert prediction_row.audit_trail.prediction.source_class == "Casting_class1"
    assert prediction_row.audit_trail.feedback.divergence == "faux_negatif"
    assert prediction_row.audit_trail.feedback.train_block_reason == "oracle_gt_defective"
    assert lot_row.lot_id == "lot_nat12"
    assert lot_row.divergences == 1


def test_nat12_airflow_task_outputs_match_pydantic_contracts() -> None:
    dataset_output = AirflowDatasetTaskOutput.model_validate(
        {
            "manifest_path": "data/candidates/manifest.csv",
            "dataset_version": "candidate_v012",
            "sample_count": 42,
            "filtered_count": 2,
            "roi_status_count": 40,
        }
    )
    train_output = AirflowTrainTaskOutput.model_validate(
        {
            "run_id": "run_nat12",
            "checkpoint": "models/feature_ae.pt",
            "run_dir": "runs/run_nat12",
        }
    )
    eval_output = AirflowEvalTaskOutput.model_validate(
        {
            "recall": 0.98,
            "ap": 0.91,
            "orange_rate": 0.12,
            "latency_ms": 18.5,
            "false_negatives": 0,
        }
    )
    gates_output = AirflowGatesTaskOutput.model_validate(
        {
            "passed": True,
            "reason": "promotion gates passed",
            "gate_results": {"recall": True, "ap": True},
        }
    )

    assert dataset_output.dataset_version == "candidate_v012"
    assert train_output.run_id == "run_nat12"
    assert eval_output.false_negatives == 0
    assert gates_output.passed is True


def test_nat12_mlflow_registry_outputs_match_pydantic_contracts() -> None:
    mlflow_output = AirflowMLflowTaskOutput.model_validate(
        {
            "registered_model_name": "feature_ae__scenario_nat12",
            "version": "12",
            "stage": "candidate",
            "run_id": "run_nat12",
        }
    )
    promotion_output = AirflowPromotionTaskOutput.model_validate(
        {
            "accepted": True,
            "registered_model_name": "feature_ae__scenario_nat12",
            "version": "12",
            "stage": "prod",
            "reason": "promotion accepted",
        }
    )

    registry_ref = ModelRegistryRef(
        scenario_id="scenario_nat12",
        registered_model_name="feature_ae__scenario_nat12",
        stage="prod",
    )
    registry_response = ModelRegistryRefResponse.model_validate(registry_ref.to_dict())

    assert mlflow_output.source_of_truth == "mlflow_registry"
    assert promotion_output.stage == ModelStage.prod
    assert registry_response.source_of_truth == "mlflow_registry"
    assert registry_response.registered_model_name == "feature_ae__scenario_nat12"
