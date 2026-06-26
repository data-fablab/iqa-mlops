from __future__ import annotations

from iqa.metadata.repository import MemoryMetadataRepository
from iqa.monitoring.lifecycle_signals import (
    collect_and_record_lifecycle_signal,
)


def _save_prediction(
    repository: MemoryMetadataRepository,
    index: int,
    *,
    scenario_id: str = "production_replay_natural",
    roi_status: str = "ok",
) -> str:
    prediction_id = f"pred_{index:03d}"
    repository.save_prediction(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "piece_event_id": f"piece_{index:03d}",
            "scenario_id": scenario_id,
            "lot_id": "lot_nat16",
            "model_version": "feature_ae_v1",
            "roi_status": roi_status,
            "created_at": f"2026-06-26T12:{index % 60:02d}:00+00:00",
        },
    )
    return prediction_id


def _save_conforming_feedback(
    repository: MemoryMetadataRepository,
    prediction_id: str,
    index: int,
) -> None:
    repository.save_feedback(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "feedback_source": "oracle_gt",
            "feedback_closed": True,
            "closed_at": f"2026-06-26T13:{index % 60:02d}:00+00:00",
            "eligible_for_train": True,
            "verdict": {"verdict": "conforme"},
        },
    )


def test_natural_signal_consumes_new_conforming_feedback_once() -> None:
    repository = MemoryMetadataRepository()

    for index in range(50):
        prediction_id = _save_prediction(repository, index)
        _save_conforming_feedback(repository, prediction_id, index)

    first = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="production_replay_natural",
    )

    assert first["trigger_lifecycle"] is True
    assert first["signal"]["conforming_validated_count"] == 50
    assert len(first["consumed_prediction_ids"]) == 50

    second = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="production_replay_natural",
    )

    assert second["trigger_lifecycle"] is False
    assert second["signal"]["conforming_validated_count"] == 0

    third = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="production_replay_natural",
    )

    assert third["lifecycle_trigger_event_id"] == second["lifecycle_trigger_event_id"]
    assert len(repository.list_lifecycle_trigger_events()) == 2


def test_non_conforming_or_ineligible_feedback_is_not_counted() -> None:
    repository = MemoryMetadataRepository()
    prediction_id = _save_prediction(repository, 1)

    repository.save_feedback(
        prediction_id,
        {
            "prediction_id": prediction_id,
            "feedback_source": "oracle_gt",
            "feedback_closed": True,
            "eligible_for_train": False,
            "verdict": {"verdict": "defective"},
        },
    )

    result = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="production_replay_natural",
    )

    assert result["signal"]["conforming_validated_count"] == 0
    assert result["trigger_lifecycle"] is False


def test_roi_fail_rate_uses_the_configured_prediction_window() -> None:
    repository = MemoryMetadataRepository()

    _save_prediction(repository, 1, roi_status="ok")
    _save_prediction(repository, 2, roi_status="fail")
    _save_prediction(repository, 3, roi_status="fail")
    _save_prediction(repository, 4, roi_status="ok")

    result = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="production_replay_natural",
        roi_window_size=4,
    )

    assert result["signal"]["roi_fail_rate"] == 0.5


def test_drift_event_triggers_only_once() -> None:
    repository = MemoryMetadataRepository()
    repository.save_scenario_version_event(
        {
            "scenario_version_id": "drift_observation_001",
            "scenario_id": "drift_domain_extension",
            "scenario_version": "drift_v001",
            "dataset_version": "drift_domain_extension_v001",
            "lifecycle_status": "drift_confirmed",
            "drift_confirmed": True,
            "created_at": "2026-06-26T14:00:00+00:00",
        }
    )

    first = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="drift_domain_extension",
    )
    second = collect_and_record_lifecycle_signal(
        repository,
        scenario_id="drift_domain_extension",
    )

    assert first["trigger_lifecycle"] is True
    assert first["consumed_drift_event_ids"] == ["drift_observation_001"]
    assert second["trigger_lifecycle"] is False
    assert second["watermark"]["drift_event_consumed"] is True
