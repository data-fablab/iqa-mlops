"""Behaviour tests for the GPU-locked train/eval boundaries (ADR 0008, issue 09).

``run_train`` and ``run_eval`` share :func:`scripts.gpu_boundary.emit_gpu_locked_summary`,
which guards the single GPU. We drive the scripts through their public interface
(argv -> JSON / exit code) and pin the concurrency contract directly on the helper.
"""

from __future__ import annotations

import json
import sys
from typing import Callable

import pytest

from iqa.runtime import gpu_lock
from scripts import run_eval, run_train
from scripts.gpu_boundary import GPU_BUSY_EXIT_CODE, emit_gpu_locked_summary


@pytest.fixture(autouse=True)
def _isolated_lock_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQA_GPU_LOCK_PATH", str(tmp_path / "gpu.lock"))


def test_emit_summary_exits_temp_fail_when_the_gpu_is_busy() -> None:
    """A held lock makes the boundary exit EX_TEMPFAIL, not crash -- so Airflow
    can tell "GPU busy" apart from a real failure."""
    with gpu_lock(owner="iqa-inference"):
        with pytest.raises(SystemExit) as exc:
            emit_gpu_locked_summary(
                owner="iqa-trainer",
                summary={"service": "iqa-trainer"},
                no_gpu_lock=False,
                wait_for_gpu=False,
            )

    assert exc.value.code == GPU_BUSY_EXIT_CODE


def test_emit_summary_skips_the_lock_in_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``no_gpu_lock`` prints the summary even while the GPU is held elsewhere."""
    with gpu_lock(owner="iqa-inference"):
        emit_gpu_locked_summary(
            owner="iqa-trainer",
            summary={"service": "iqa-trainer", "status": "validated"},
            no_gpu_lock=True,
            wait_for_gpu=False,
        )

    assert json.loads(capsys.readouterr().out)["status"] == "validated"


def test_run_train_emits_validated_summary_under_the_lock(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    result = run_boundary_script(
        run_train,
        ["iqa-run-train", "--scenario-id", "s1", "--dataset-version", "v1"],
    )

    assert result["service"] == "iqa-trainer"
    assert result["stage"] == "train"
    assert result["status"] == "validated"
    assert result["persisted"] is False


def test_run_train_carries_the_dataset_uri_into_its_summary(
    run_boundary_script: Callable[[object, list[str]], dict],
) -> None:
    """The candidate dataset URI from the dataset task is traceable in train's XCom."""
    uri = "s3://iqa-source-datasets/model_datasets/s1/v1/candidate.csv"

    result = run_boundary_script(
        run_train,
        ["iqa-run-train", "--scenario-id", "s1", "--dataset-uri", uri],
    )

    assert result["dataset_uri"] == uri


def test_train_resolves_a_dataset_materialised_by_the_dataset_boundary(
    tmp_path,
) -> None:
    """Producer->consumer loop: train reads back what the dataset task wrote (issue 20)."""
    from iqa.storage.object_store import InMemoryObjectStore

    from scripts import run_dataset

    manifest = tmp_path / "candidate.csv"
    manifest.write_bytes(b"event_id,scenario_id\nevt_1,s1\nevt_2,s1\n")
    store = InMemoryObjectStore()
    uri = run_dataset.materialise_dataset(
        store, manifest=manifest, scenario_id="s1", candidate_version="v1"
    )

    resolved = run_train.resolve_dataset(store, uri)

    assert resolved["uri"] == uri
    assert resolved["rows"] == 2


def test_train_resolution_raises_when_the_dataset_uri_is_absent() -> None:
    from iqa.storage.object_store import InMemoryObjectStore

    with pytest.raises(KeyError, match="s3://iqa-source-datasets/missing.csv"):
        run_train.resolve_dataset(
            InMemoryObjectStore(), "s3://iqa-source-datasets/missing.csv"
        )


def test_run_eval_refuses_when_the_gpu_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--wait-for-gpu`` a busy GPU is refused immediately (exit 75)."""
    monkeypatch.setattr(sys, "argv", ["iqa-run-eval", "--scenario-id", "s1"])

    with gpu_lock(owner="iqa-trainer"):
        with pytest.raises(SystemExit) as exc:
            run_eval.main()

    assert exc.value.code == GPU_BUSY_EXIT_CODE
