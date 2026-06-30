"""Module-level contract for the observability exposition seam.

These tests exercise sanitisation and Prometheus rendering directly on
``ObservabilityExposition`` — no FastAPI ``TestClient``, no HTTP. The seam's
interface (record + render + reject) is the test surface, which is the point of
extracting the subsystem out of ``iqa.api.main``.
"""

from __future__ import annotations

import pytest

from iqa.api.schemas import DriftEventRequest, LifecycleEventRequest
from iqa.monitoring.exposition import ObservabilityExposition, ObservabilityRejection


@pytest.fixture
def exposition() -> ObservabilityExposition:
    return ObservabilityExposition()


def test_lifecycle_event_renders_epoch_metrics(exposition: ObservabilityExposition) -> None:
    exposition.record_lifecycle_event(
        LifecycleEventRequest(
            event_type="epoch_completed",
            scenario_id="production_replay_natural_piece_b_full",
            lifecycle_run_id="replay_lifecycle_001",
            cycle_id="cycle_1",
            epoch=3,
            metrics={"pixel_aupimo": 0.81, "image_ap": 0.74},
        )
    )

    lines = exposition.render_prometheus_lines()
    rendered = "\n".join(lines)

    assert "iqa_lifecycle_epoch_current{" in rendered
    assert 'metric="pixel_aupimo"' in rendered
    assert "iqa_lifecycle_epoch_pixel_aupimo{" in rendered
    assert "iqa_lifecycle_epoch_image_ap{" in rendered


def test_lifecycle_event_rejects_path_like_value(exposition: ObservabilityExposition) -> None:
    with pytest.raises(ObservabilityRejection) as exc_info:
        exposition.record_lifecycle_event(
            LifecycleEventRequest(
                event_type="epoch_completed",
                scenario_id="production_replay_natural_piece_b_full",
                lifecycle_run_id="replay_lifecycle_001",
                candidate_version="/models/leaked_checkpoint",
            )
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == "sensitive_lifecycle_value"


def test_lifecycle_event_rejects_unsupported_metric(exposition: ObservabilityExposition) -> None:
    with pytest.raises(ObservabilityRejection) as exc_info:
        exposition.record_lifecycle_event(
            LifecycleEventRequest(
                event_type="epoch_completed",
                scenario_id="production_replay_natural_piece_b_full",
                lifecycle_run_id="replay_lifecycle_001",
                metrics={"unsupported_metric": 1.0},
            )
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == "unsupported_lifecycle_metric"


def test_drift_event_renders_status_and_score(exposition: ObservabilityExposition) -> None:
    exposition.record_drift_event(
        DriftEventRequest(
            event_type="window_evaluated",
            scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
            status="confirmed",
            metrics={"drift_score": 0.62, "red_rate": 0.3},
        )
    )

    rendered = "\n".join(exposition.render_prometheus_lines())

    assert 'iqa_drift_status{scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",source_domain="piece_a_p4",status="confirmed"} 1' in rendered
    assert "iqa_drift_score{" in rendered


def test_drift_event_rejects_unsupported_metric(exposition: ObservabilityExposition) -> None:
    with pytest.raises(ObservabilityRejection) as exc_info:
        exposition.record_drift_event(
            DriftEventRequest(
                event_type="window_evaluated",
                scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
                metrics={"not_a_drift_metric": 1.0},
            )
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.error_code == "unsupported_drift_metric"


def test_reset_clears_accumulated_state(exposition: ObservabilityExposition) -> None:
    exposition.record_drift_event(
        DriftEventRequest(
            event_type="window_evaluated",
            scenario_id="production_replay_natural_piece_b_to_piece_a_p4_drift",
            status="confirmed",
            metrics={"drift_score": 0.62},
        )
    )
    assert "iqa_drift_score{" in "\n".join(exposition.render_prometheus_lines())

    exposition.reset()

    # Only HELP/TYPE headers remain; no observed drift series.
    assert "iqa_drift_score{" not in "\n".join(exposition.render_prometheus_lines())
