from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import run_ingestion, run_monitoring, run_replay


def _run_script(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], module: object, args: list[str]) -> dict:
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    return json.loads(capsys.readouterr().out)


def test_run_ingestion_validates_manifest_and_reports_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "pieces.csv"
    manifest.write_text(
        "event_id,scenario_id,dataset_version,source_class\n"
        "evt_1,raw_ingestion,hss_iad_casting_raw_v1,Casting_class1\n",
        encoding="utf-8",
    )

    result = _run_script(
        monkeypatch,
        capsys,
        run_ingestion,
        ["iqa-run-ingestion", "--manifest", str(manifest), "--scenario-id", "raw_ingestion"],
    )

    assert result["status"] == "validated"
    assert result["service"] == "iqa-ingestion"
    assert result["manifest"]["row_count"] == 1
    assert result["manifest"]["dataset_versions"] == ["hss_iad_casting_raw_v1"]


def test_run_ingestion_fails_clearly_for_missing_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["iqa-run-ingestion", "--manifest", "missing.csv"])

    with pytest.raises(FileNotFoundError, match="ingestion manifest not found"):
        run_ingestion.main()


def test_run_replay_validates_plan_for_requested_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = tmp_path / "replay.csv"
    plan.write_text(
        "simulated_event_id,scenario_id,dataset_version,lot_id,source_class\n"
        "sim_1,production_replay_natural,production_replay_natural_v001,IQA-001,Casting_class1\n",
        encoding="utf-8",
    )

    result = _run_script(
        monkeypatch,
        capsys,
        run_replay,
        ["iqa-run-replay", "--scenario-id", "production_replay_natural", "--plan", str(plan)],
    )

    assert result["status"] == "validated"
    assert result["plan_event_count"] == 1
    assert result["dataset_versions"] == ["production_replay_natural_v001"]
    assert result["lot_ids"] == ["IQA-001"]


def test_run_replay_rejects_unknown_scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = tmp_path / "replay.csv"
    plan.write_text("scenario_id\nunknown_scenario\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["iqa-run-replay", "--scenario-id", "unknown_scenario", "--plan", str(plan)])

    with pytest.raises(ValueError, match="unknown replay scenario_id"):
        run_replay.main()


def test_run_monitoring_reports_lifecycle_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _run_script(
        monkeypatch,
        capsys,
        run_monitoring,
        ["iqa-run-monitoring", "--scenario-id", "production_replay_natural", "--conforming-validated-count", "50"],
    )

    assert result["status"] == "validated"
    assert result["trigger_lifecycle"] is True
    assert result["lifecycle_decision"]["trigger_reason"] == "natural_50_oracle_conformes"
