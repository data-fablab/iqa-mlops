import json
from pathlib import Path

import pytest

from scripts.lineage_summary import build_lineage_summary

ROOT = Path(__file__).resolve().parents[2]


def _write_replay_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "scenario_id": "production_replay_natural",
                "run_id": "replay_lifecycle_test",
                "mode": "decision-only",
                "events_processed": 2,
                "lots_processed": 1,
                "trigger_lifecycle": False,
                "trigger_reason": "natural_waiting_for_50_oracle_conformes",
                "candidate_dataset_version": "",
                "candidate_checkpoint": None,
                "mlflow_run_id": None,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "piece_event_id": "evt-001",
                        "lot_id": "lot-001",
                        "scenario_id": "production_replay_natural",
                        "dataset_version": "production_replay_natural_v001",
                        "source_class": "Casting_class1",
                        "threshold_source": "manifest:calibration_good_quantiles",
                    }
                ),
                json.dumps(
                    {
                        "piece_event_id": "evt-002",
                        "lot_id": "lot-001",
                        "scenario_id": "production_replay_natural",
                        "dataset_version": "production_replay_natural_v001",
                        "source_class": "Casting_class1",
                        "threshold_source": "manifest:calibration_good_quantiles",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "lots.jsonl").write_text(
        json.dumps(
            {
                "lot_id": "lot-001",
                "scenario_id": "production_replay_natural",
                "dataset_versions": ["production_replay_natural_v001"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_lineage_doc_covers_phase3_evidence_surfaces() -> None:
    content = (ROOT / "docs" / "lineage-evidence.md").read_text(encoding="utf-8")

    for expected in [
        "piece_event_id -> lot_id -> scenario_id -> dataset_version -> manifest_version",
        "DVC/MinIO",
        "MLflow Registry",
        "PostgreSQL metadata",
        "iqa-lineage-summary",
        "iqa-check-dvc-reproducibility --with-network",
        "Marc",
        "Laurent",
    ]:
        assert expected in content


def test_lineage_summary_command_is_public() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'iqa-lineage-summary = "scripts.lineage_summary:main"' in pyproject


def test_lineage_summary_builds_complete_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_replay_run(run_dir)

    summary = build_lineage_summary(
        replay_run_dir=run_dir,
        model_version="rd_feature_ae_gated_v001_bootstrap",
        dvc_yaml=ROOT / "dvc.yaml",
    )

    assert summary["lineage_status"] == "complete"
    assert summary["scenario_id"] == "production_replay_natural"
    assert summary["dataset_versions"] == [
        "production_replay_natural_v001",
        "feature_ae_good_v001_bootstrap",
    ]
    assert summary["lot_ids"] == ["lot-001"]
    assert summary["model_artifact_uri"] == "s3://iqa-models/rd_feature_ae_gated_v001_bootstrap/checkpoint.pt"
    assert summary["model_sha256"]
    assert summary["preprocessing_contract_version"] == "feature_ae_preprocessing_v001"
    assert summary["decision_thresholds"]["calibration_set_id"] == "calibration_set_v001"
    assert summary["threshold_sources"] == ["manifest:calibration_good_quantiles"]
    assert set(summary["dvc"]["stages"]) >= {"inventory", "piece_events", "replay", "validation", "model_dataset"}
    assert summary["mlflow_tracking"]["source_of_truth"] == "mlflow_registry"


def test_lineage_summary_rejects_incomplete_replay_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="events.jsonl"):
        build_lineage_summary(
            replay_run_dir=run_dir,
            model_version="rd_feature_ae_gated_v001_bootstrap",
            dvc_yaml=ROOT / "dvc.yaml",
        )


def test_mlflow_logger_declares_required_lineage_fields() -> None:
    source = (ROOT / "src" / "iqa" / "training" / "mlflow_logging.py").read_text(encoding="utf-8")

    for expected in [
        '"dataset_version"',
        '"manifest_version"',
        '"git_commit"',
        '"scenario_id"',
        '"preprocessing_contract_version"',
        "mlflow.set_tags(tags)",
    ]:
        assert expected in source
