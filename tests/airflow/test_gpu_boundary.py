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


def test_run_eval_refuses_when_the_gpu_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``--wait-for-gpu`` a busy GPU is refused immediately (exit 75)."""
    monkeypatch.setattr(sys, "argv", ["iqa-run-eval", "--scenario-id", "s1"])

    with gpu_lock(owner="iqa-trainer"):
        with pytest.raises(SystemExit) as exc:
            run_eval.main()

    assert exc.value.code == GPU_BUSY_EXIT_CODE
