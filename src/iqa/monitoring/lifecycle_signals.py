"""Collect durable lifecycle signals from the metadata repository."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from iqa.metadata.repository import MetadataRepository
from iqa.monitoring.lifecycle import LifecycleSignal, evaluate_lifecycle_signal


def _is_conforming_oracle_feedback(record: dict[str, Any] | None) -> bool:
    if not record:
        return False

    verdict = record.get("verdict")
    if isinstance(verdict, dict):
        verdict = verdict.get("verdict")

    return (
        record.get("feedback_source") == "oracle_gt"
        and record.get("feedback_closed") is True
        and record.get("eligible_for_train") is True
        and verdict == "conforme"
    )


def _consumed_sources(
    repository: MetadataRepository,
    scenario_id: str,
) -> tuple[set[str], set[str]]:
    prediction_ids: set[str] = set()
    drift_event_ids: set[str] = set()

    for event in repository.list_lifecycle_trigger_events():
        if event.get("scenario_id") != scenario_id:
            continue
        if event.get("trigger_lifecycle") is not True:
            continue

        prediction_ids.update(event.get("consumed_prediction_ids") or [])
        drift_event_ids.update(event.get("consumed_drift_event_ids") or [])

    return prediction_ids, drift_event_ids


def _natural_candidates(
    repository: MetadataRepository,
    predictions: list[dict[str, Any]],
    scenario_id: str,
    consumed_prediction_ids: set[str],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for prediction in predictions:
        prediction_id = str(prediction.get("prediction_id") or "")
        if not prediction_id:
            continue
        if prediction.get("scenario_id") != scenario_id:
            continue
        if prediction_id in consumed_prediction_ids:
            continue

        feedback = repository.get_feedback(prediction_id)
        if not _is_conforming_oracle_feedback(feedback):
            continue

        candidates.append(
            {
                "prediction_id": prediction_id,
                "closed_at": (feedback or {}).get("closed_at"),
                "lot_id": prediction.get("lot_id"),
                "model_version": prediction.get("model_version"),
            }
        )

    candidates.sort(
        key=lambda item: (
            str(item.get("closed_at") or ""),
            item["prediction_id"],
        )
    )
    return candidates


def _roi_window(
    predictions: list[dict[str, Any]],
    scenario_id: str,
    window_size: int,
) -> tuple[list[dict[str, Any]], float]:
    matching = [
        prediction
        for prediction in predictions
        if prediction.get("scenario_id") == scenario_id
    ]
    matching.sort(
        key=lambda item: (
            str(item.get("created_at") or ""),
            str(item.get("prediction_id") or ""),
        ),
        reverse=True,
    )

    window = matching[: max(int(window_size), 1)]
    roi_fail_count = sum(
        1
        for prediction in window
        if prediction.get("roi_status") in {"fail", "roi_fail"}
    )
    rate = roi_fail_count / len(window) if window else 0.0
    return window, rate


def _latest_drift_event(
    repository: MetadataRepository,
    scenario_id: str,
) -> dict[str, Any] | None:
    matching = [
        event
        for event in repository.list_scenario_version_events()
        if event.get("scenario_id") == scenario_id
    ]
    return matching[-1] if matching else None


def _event_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"lifecycle_trigger_{hashlib.sha256(encoded).hexdigest()[:24]}"


def collect_and_record_lifecycle_signal(
    repository: MetadataRepository,
    *,
    scenario_id: str,
    roi_window_size: int = 100,
    min_natural_conforming: int = 50,
) -> dict[str, Any]:
    """Collect persisted signals, evaluate the rule and journal the decision."""

    predictions = repository.list_predictions()
    consumed_predictions, consumed_drift_events = _consumed_sources(
        repository,
        scenario_id,
    )

    natural_candidates = _natural_candidates(
        repository,
        predictions,
        scenario_id,
        consumed_predictions,
    )
    roi_predictions, roi_fail_rate = _roi_window(
        predictions,
        scenario_id,
        roi_window_size,
    )

    drift_event = _latest_drift_event(repository, scenario_id)
    drift_event_id = (
        str(drift_event.get("scenario_version_id"))
        if drift_event and drift_event.get("scenario_version_id")
        else None
    )
    drift_was_consumed = (
        drift_event_id in consumed_drift_events
        if drift_event_id is not None
        else False
    )
    drift_confirmed = bool(
        drift_event
        and not drift_was_consumed
        and (
            drift_event.get("drift_confirmed") is True
            or drift_event.get("lifecycle_status") == "drift_confirmed"
        )
    )

    signal = LifecycleSignal(
        scenario_id=scenario_id,
        conforming_validated_count=len(natural_candidates),
        drift_confirmed=drift_confirmed,
        roi_fail_rate=roi_fail_rate,
    )
    decision = evaluate_lifecycle_signal(
        signal,
        min_natural_conforming=min_natural_conforming,
    )

    consumed_prediction_ids = (
        [item["prediction_id"] for item in natural_candidates]
        if decision.trigger_lifecycle
        else []
    )
    consumed_drift_event_ids = (
        [drift_event_id]
        if decision.trigger_lifecycle and drift_event_id is not None
        else []
    )

    watermark = {
        "latest_feedback_closed_at": (
            natural_candidates[-1].get("closed_at")
            if natural_candidates
            else None
        ),
        "latest_prediction_id": (
            natural_candidates[-1]["prediction_id"]
            if natural_candidates
            else None
        ),
        "drift_event_id": drift_event_id,
        "drift_event_consumed": drift_was_consumed,
    }
    fingerprint = {
        "scenario_id": scenario_id,
        "candidate_prediction_ids": [
            item["prediction_id"] for item in natural_candidates
        ],
        "drift_event_id": drift_event_id,
        "drift_event_consumed": drift_was_consumed,
        "roi_window_prediction_ids": [
            prediction.get("prediction_id") for prediction in roi_predictions
        ],
        "trigger_lifecycle": decision.trigger_lifecycle,
    }
    lifecycle_trigger_event_id = _event_id(fingerprint)
    created_at = datetime.now(timezone.utc).isoformat()

    event = {
        "lifecycle_trigger_event_id": lifecycle_trigger_event_id,
        "scenario_id": scenario_id,
        "trigger_reason": decision.trigger_reason,
        "trigger_lifecycle": decision.trigger_lifecycle,
        "dataset_version": decision.candidate_dataset_version,
        "manifest_version": None,
        "model_version": (
            natural_candidates[-1].get("model_version")
            if natural_candidates
            else None
        ),
        "lot_id": (
            natural_candidates[-1].get("lot_id")
            if natural_candidates
            else None
        ),
        "signal": signal.to_dict(),
        "watermark": watermark,
        "consumed_prediction_ids": consumed_prediction_ids,
        "consumed_drift_event_ids": consumed_drift_event_ids,
        "roi_window_size": len(roi_predictions),
        "created_at": created_at,
    }
    repository.save_lifecycle_trigger_event(event)

    return {
        "service": "iqa-lifecycle-signal-collector",
        "signal": signal.to_dict(),
        "lifecycle_decision": decision.to_dict(),
        "trigger_lifecycle": decision.trigger_lifecycle,
        "lifecycle_trigger_event_id": lifecycle_trigger_event_id,
        "watermark": watermark,
        "consumed_prediction_ids": consumed_prediction_ids,
        "consumed_drift_event_ids": consumed_drift_event_ids,
        "status": "validated",
    }


__all__ = ["collect_and_record_lifecycle_signal"]
