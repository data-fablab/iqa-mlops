from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

from scripts import run_ingestion, run_monitoring, run_replay


def test_run_ingestion_materialises_manifest_and_reports_counts(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """The ingestion boundary writes the manifest to the object store (issue 18)."""
    manifest = tmp_path / "pieces.csv"
    manifest.write_text(
        "event_id,scenario_id,dataset_version,source_class\n"
        "evt_1,raw_ingestion,hss_iad_casting_raw_v1,Casting_class1\n",
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_ingestion,
        ["iqa-run-ingestion", "--manifest", str(manifest), "--scenario-id", "raw_ingestion"],
    )

    assert result["status"] == "ingested"
    assert result["service"] == "iqa-ingestion"
    assert result["manifest"]["row_count"] == 1
    assert result["manifest"]["dataset_versions"] == ["hss_iad_casting_raw_v1"]
    assert result["materialized"] is True
    assert result["ingested_uri"].startswith("s3://")


def test_materialise_ingestion_writes_exact_bytes_to_a_deterministic_key(
    tmp_path: Path,
) -> None:
    """The ingested manifest lands verbatim at a scenario/source-derived key."""
    from iqa.storage import IQA_BUCKETS, parse_s3_uri
    from iqa.storage.object_store import InMemoryObjectStore

    manifest = tmp_path / "pieces.csv"
    body = b"event_id,scenario_id\nevt_1,raw_ingestion\n"
    manifest.write_bytes(body)
    store = InMemoryObjectStore()

    uri = run_ingestion.materialise_ingestion(
        store,
        manifest=manifest,
        scenario_id="raw_ingestion",
        source="historical_replay",
    )

    parsed = parse_s3_uri(uri)
    assert parsed.bucket == IQA_BUCKETS["ingested_images"]
    assert parsed.key == "ingested/raw_ingestion/historical_replay/pieces.csv"
    assert store.get_bytes(parsed.bucket, parsed.key) == body


def test_run_ingestion_fails_clearly_for_missing_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["iqa-run-ingestion", "--manifest", "missing.csv"])

    with pytest.raises(FileNotFoundError, match="ingestion manifest not found"):
        run_ingestion.main()


def test_run_replay_validates_plan_for_requested_scenario(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    plan = tmp_path / "replay.csv"
    plan.write_text(
        "simulated_event_id,scenario_id,dataset_version,lot_id,source_class\n"
        "sim_1,production_replay_natural,production_replay_natural_v001,IQA-001,Casting_class1\n",
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_replay,
        ["iqa-run-replay", "--scenario-id", "production_replay_natural", "--plan", str(plan)],
    )

    assert result["status"] == "validated"
    assert result["plan_event_count"] == 1
    assert result["dataset_versions"] == ["production_replay_natural_v001"]
    assert result["lot_ids"] == ["IQA-001"]


def test_run_replay_preserves_event_time_recorded_at_is_simulated(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    plan = tmp_path / "replay.csv"
    plan.write_text(
        "simulated_event_id,scenario_id,lot_id,event_time,recorded_at,is_simulated\n"
        "sim_1,production_replay_natural,IQA-001,2026-01-02T08:00:00,2026-01-02T08:00:05,true\n",
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_replay,
        ["iqa-run-replay", "--scenario-id", "production_replay_natural", "--plan", str(plan)],
    )

    # Replayed events keep their temporal/simulation semantics (acceptance criterion).
    assert result["preserved_event_fields"] == ["event_time", "recorded_at", "is_simulated"]
    assert result["is_simulated_values"] == ["true"]


def test_run_replay_rejects_unknown_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "replay.csv"
    plan.write_text("scenario_id\nunknown_scenario\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["iqa-run-replay", "--scenario-id", "unknown_scenario", "--plan", str(plan)])

    with pytest.raises(ValueError, match="unknown replay scenario_id"):
        run_replay.main()


def test_run_monitoring_reports_lifecycle_decision(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    result = run_boundary_script(
        run_monitoring,
        ["iqa-run-monitoring", "--scenario-id", "production_replay_natural", "--conforming-validated-count", "50"],
    )

    assert result["status"] == "validated"
    assert result["trigger_lifecycle"] is True
    assert result["lifecycle_decision"]["trigger_reason"] == "natural_50_oracle_conformes"


def test_run_monitoring_evaluates_thresholds_config_in_container(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    thresholds = tmp_path / "monitoring_thresholds.yaml"
    thresholds.write_text(
        "quality:\n  roi_fail_rate_warning: 0.05\n  roi_fail_rate_critical: 0.10\n",
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_monitoring,
        [
            "iqa-run-monitoring",
            "--scenario-id", "production_replay_natural",
            "--roi-fail-rate", "0.12",
            "--thresholds-config", str(thresholds),
        ],
    )

    # ROI fail rate above the critical threshold is flagged in-container.
    assert result["thresholds_evaluated"] is True
    roi = result["roi_fail_rate_evaluation"]
    assert roi["status"] == "critical"
    assert roi["breached"] is True
    assert roi["critical"] == 0.10


def test_run_monitoring_detects_piece_a_p4_drift_after_confirmed_windows(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    thresholds = tmp_path / "monitoring_thresholds.yaml"
    thresholds.write_text(
        "\n".join(
            [
                "drift:",
                "  min_window_events: 30",
                "  confirm_windows: 2",
                "  domain_ratio_critical: 0.50",
                "  alert_rate_critical: 0.50",
                "  red_rate_critical: 0.20",
                "  unexpected_red_rate_critical: 0.20",
                "  roi_fail_rate_critical: 0.10",
                "  oracle_fn_rate_critical: 0.05",
                "",
                "quality:",
                "  roi_fail_rate_warning: 0.05",
                "  roi_fail_rate_critical: 0.10",
            ]
        ),
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_monitoring,
        [
            "iqa-run-monitoring",
            "--scenario-id",
            "production_replay_natural_piece_b_to_piece_a_p4_drift",
            "--window-events",
            "60",
            "--domain-ratio",
            "0.70",
            "--alert-rate",
            "0.70",
            "--unexpected-red-rate",
            "0.70",
            "--critical-window-count",
            "1",
            "--thresholds-config",
            str(thresholds),
        ],
    )

    assert result["drift_suspected"] is True
    assert result["drift_confirmed"] is True
    assert result["trigger_lifecycle"] is True
    assert result["trigger_reason"] == "drift_piece_a_p4_confirmed"


def test_run_monitoring_does_not_confirm_piece_a_p4_on_domain_ratio_only(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    thresholds = tmp_path / "monitoring_thresholds.yaml"
    thresholds.write_text(
        "\n".join(
            [
                "drift:",
                "  min_window_events: 30",
                "  confirm_windows: 2",
                "  domain_ratio_critical: 0.50",
                "  alert_rate_critical: 0.50",
                "  red_rate_critical: 0.20",
                "  unexpected_red_rate_critical: 0.20",
                "  roi_fail_rate_critical: 0.10",
                "  oracle_fn_rate_critical: 0.05",
            ]
        ),
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_monitoring,
        [
            "iqa-run-monitoring",
            "--scenario-id",
            "production_replay_natural_piece_b_to_piece_a_p4_drift",
            "--window-events",
            "60",
            "--domain-ratio",
            "0.90",
            "--critical-window-count",
            "99",
            "--thresholds-config",
            str(thresholds),
        ],
    )

    assert result["drift_suspected"] is True
    assert result["drift_confirmed"] is False
    assert result["trigger_lifecycle"] is False
    assert result["drift_evaluation"]["signals"]["domain_ratio"] is True
    assert result["drift_evaluation"]["critical_window"] is False


def test_run_monitoring_does_not_confirm_drift_on_small_window(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    thresholds = tmp_path / "monitoring_thresholds.yaml"
    thresholds.write_text("drift:\n  min_window_events: 30\n  confirm_windows: 2\n", encoding="utf-8")

    result = run_boundary_script(
        run_monitoring,
        [
            "iqa-run-monitoring",
            "--scenario-id",
            "production_replay_natural_piece_b_to_piece_a_p4_drift",
            "--window-events",
            "10",
            "--domain-ratio",
            "0.90",
            "--critical-window-count",
            "2",
            "--thresholds-config",
            str(thresholds),
        ],
    )

    assert result["drift_suspected"] is False
    assert result["drift_confirmed"] is False
    assert result["trigger_lifecycle"] is False
