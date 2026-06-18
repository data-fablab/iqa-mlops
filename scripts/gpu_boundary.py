"""Shared GPU-locked entrypoint for the train/eval boundaries (ADR 0008, issue 09).

``train`` and ``eval`` run on the ml image and must hold the single-GPU lock for
their whole duration (shared with the inference service) so only one GPU consumer
runs at a time. They differ only in their summary payload, so the lock dance --
including the ``EX_TEMPFAIL`` exit code that lets Airflow tell "GPU busy" apart
from a real failure -- lives here, in one place.

This module imports ``iqa.runtime`` (the lock) on purpose: it is only imported by
the GPU boundary scripts, never by the lighter data-image boundaries.
"""

from __future__ import annotations

import sys
from typing import Any

from iqa.runtime import GpuBusyError, gpu_lock
from scripts.airflow_contracts import print_json

# BSD EX_TEMPFAIL: emitted when the single GPU is already held, so the
# orchestrator can distinguish "GPU busy" from a genuine task failure.
GPU_BUSY_EXIT_CODE = 75


def emit_gpu_locked_summary(
    *,
    owner: str,
    summary: dict[str, Any],
    no_gpu_lock: bool,
    wait_for_gpu: bool,
) -> None:
    """Print ``summary`` while holding the single-GPU lock for ``owner``.

    ``no_gpu_lock`` skips the lock entirely (CPU-only dry run). When the lock is
    held by another consumer, exit :data:`GPU_BUSY_EXIT_CODE` instead of crashing.
    """
    if no_gpu_lock:
        print_json(summary)
        return
    try:
        with gpu_lock(owner=owner, blocking=wait_for_gpu):
            print_json(summary)
    except GpuBusyError as exc:
        print(f"{owner}: {exc}", file=sys.stderr)
        raise SystemExit(GPU_BUSY_EXIT_CODE) from exc


__all__ = ["GPU_BUSY_EXIT_CODE", "emit_gpu_locked_summary"]
