"""Behaviour tests for the iqa_lifecycle boundary scripts (ADR 0008).

These scripts are the container entrypoints behind each lifecycle task. We drive
them through their public interface -- argv in, JSON on stdout, exit code -- the
exact contract Airflow relies on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

from scripts import (
    run_dataset,
    run_gates,
    run_lifecycle_decision,
    run_mlflow,
    run_promotion,
    run_reload,
)


def test_run_gates_blocks_the_dag_when_a_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing candidate must exit non-zero so mlflow/promotion never run."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["iqa-run-gates", "--recall", "0.0", "--ap", "0.0"],
    )

    with pytest.raises(SystemExit) as exc:
        run_gates.main()

    assert exc.value.code == 1


@pytest.mark.parametrize(
    ("drift_value", "expected_trigger"),
    [("true", True), ("false", False), ("1", True), ("no", False)],
)
def test_run_lifecycle_decision_parses_drift_confirmed_as_a_value(
    drift_value: str,
    expected_trigger: bool,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """``--drift-confirmed`` arrives as a templated string and flips the decision.

    In the drift scenario the rule triggers iff drift is confirmed, so this also
    pins the str2bool parsing the Airflow argv depends on.
    """
    result = run_boundary_script(
        run_lifecycle_decision,
        [
            "iqa-run-lifecycle-decision",
            "--scenario-id", "drift_domain_extension",
            "--drift-confirmed", drift_value,
        ],
    )

    assert result["trigger_lifecycle"] is expected_trigger
    assert result["signal"]["drift_confirmed"] is expected_trigger


def test_run_lifecycle_decision_triggers_on_enough_natural_conformes(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """50 oracle-validated conformes is the natural-scenario trigger threshold."""
    result = run_boundary_script(
        run_lifecycle_decision,
        [
            "iqa-run-lifecycle-decision",
            "--scenario-id", "production_replay_natural",
            "--conforming-validated-count", "50",
        ],
    )

    assert result["trigger_lifecycle"] is True
    assert result["lifecycle_decision"]["trigger_reason"] == "natural_50_oracle_conformes"


def test_run_gates_honours_the_recall_threshold_from_its_config(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """A recall the default gate (1.0) would reject passes once the config relaxes it.

    This proves the YAML config is loaded in-container and its thresholds drive
    the decision, not just the hard-coded defaults.
    """
    config = tmp_path / "promotion_gates.yaml"
    config.write_text(
        "feature_ae:\n  recall_defect_min: 0.9\n",
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_gates,
        [
            "iqa-run-gates",
            "--recall", "0.95",
            "--gates-config", str(config),
        ],
    )

    assert result["status"] == "validated"
    assert result["all_passed"] is True
    assert result["gates"]["gates"]["recall"]["threshold"] == 0.9


def test_run_gates_uses_default_recall_threshold_without_a_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing config falls back to the strict default (recall must be 1.0)."""
    missing = tmp_path / "absent.yaml"

    monkeypatch.setattr(
        sys,
        "argv",
        ["iqa-run-gates", "--recall", "0.95", "--gates-config", str(missing)],
    )

    with pytest.raises(SystemExit) as exc:
        run_gates.main()

    assert exc.value.code == 1


def test_run_mlflow_resolves_the_scenario_isolated_model_name(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """ADR 0006: the registered model name is isolated per scenario_id."""
    result = run_boundary_script(
        run_mlflow,
        ["iqa-run-mlflow", "--scenario-id", "drift_domain_extension"],
    )

    assert result["registered_model_name"] == "feature_ae__drift_domain_extension"
    # Boundary only: no real Registry write yet (issue 21).
    assert result["registered"] is False


def test_run_promotion_snapshots_prod_only_on_a_prod_target(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """A prod promotion first snapshots prod (rollback safety); test does not."""
    to_prod = run_boundary_script(
        run_promotion,
        ["iqa-run-promotion", "--source-stage", "candidate", "--target-stage", "prod"],
    )
    to_test = run_boundary_script(
        run_promotion,
        ["iqa-run-promotion", "--source-stage", "candidate", "--target-stage", "test"],
    )

    assert to_prod["snapshot_previous_prod"] is True
    assert to_prod["transition"] == {"from": "candidate", "to": "prod"}
    assert to_test["snapshot_previous_prod"] is False


def test_run_reload_skips_unless_the_target_is_prod(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """Inference reload only fires for prod promotions; test leaves prod untouched."""
    to_test = run_boundary_script(
        run_reload,
        ["iqa-run-reload", "--target-stage", "test"],
    )
    to_prod = run_boundary_script(
        run_reload,
        ["iqa-run-reload", "--scenario-id", "production_replay_natural", "--target-stage", "prod"],
    )

    assert to_test["status"] == "skipped"
    assert to_test["reloaded"] is False
    assert to_prod["status"] == "validated"
    assert to_prod["registered_model_name"] == "feature_ae__production_replay_natural"


def test_run_dataset_materialises_the_candidate_and_reports_its_uri(
    tmp_path: Path,
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """The dataset boundary writes the candidate to the object store (issue 19)."""
    manifest = tmp_path / "candidate.csv"
    manifest.write_text(
        "event_id,scenario_id,dataset_version\n"
        "evt_1,production_replay_natural,feature_ae_good_v002\n",
        encoding="utf-8",
    )

    result = run_boundary_script(
        run_dataset,
        ["iqa-run-dataset", "--manifest", str(manifest), "--candidate-version", "v002"],
    )

    assert result["status"] == "materialized"
    assert result["manifest"]["row_count"] == 1
    assert result["candidate_version"] == "v002"
    assert result["materialized"] is True
    assert result["dataset_uri"].startswith("s3://")


def test_run_dataset_emits_the_dataset_uri_as_its_last_stdout_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """DockerOperator pushes the last stdout line as XCom: it must be the URI."""
    manifest = tmp_path / "candidate.csv"
    manifest.write_text("event_id\nevt_1\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["iqa-run-dataset", "--manifest", str(manifest),
         "--scenario-id", "s1", "--candidate-version", "v002"],
    )

    run_dataset.main()

    last_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert last_line == "s3://iqa-source-datasets/model_datasets/s1/v002/candidate.csv"


def test_materialise_dataset_writes_exact_bytes_to_a_deterministic_key(
    tmp_path: Path,
) -> None:
    """The candidate bytes land verbatim at a scenario/version-derived key."""
    from iqa.storage import IQA_BUCKETS, parse_s3_uri
    from iqa.storage.object_store import InMemoryObjectStore

    manifest = tmp_path / "candidate.csv"
    body = b"event_id,scenario_id\nevt_1,production_replay_natural\n"
    manifest.write_bytes(body)
    store = InMemoryObjectStore()

    uri = run_dataset.materialise_dataset(
        store,
        manifest=manifest,
        scenario_id="production_replay_natural",
        candidate_version="v002",
    )

    parsed = parse_s3_uri(uri)
    assert parsed.bucket == IQA_BUCKETS["source_datasets"]
    assert parsed.key == "model_datasets/production_replay_natural/v002/candidate.csv"
    assert store.get_bytes(parsed.bucket, parsed.key) == body


def test_materialise_dataset_falls_back_to_candidate_segment_without_a_version(
    tmp_path: Path,
) -> None:
    from iqa.storage.object_store import InMemoryObjectStore

    manifest = tmp_path / "candidate.csv"
    manifest.write_bytes(b"event_id\nevt_1\n")
    store = InMemoryObjectStore()

    uri = run_dataset.materialise_dataset(
        store,
        manifest=manifest,
        scenario_id="production_replay_natural",
        candidate_version="",
    )

    assert uri.endswith("model_datasets/production_replay_natural/candidate/candidate.csv")


def test_run_dataset_fails_clearly_for_a_missing_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["iqa-run-dataset", "--manifest", "missing.csv"])

    with pytest.raises(FileNotFoundError, match="dataset manifest not found"):
        run_dataset.main()
