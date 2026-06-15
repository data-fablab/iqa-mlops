"""Internal helpers for inference processing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import torch


@contextmanager
def measure_inference_time() -> Generator[dict[str, float], None, None]:
    """Context manager that measures elapsed time during inference.

    Yields a dict with 'start' key for internal use, stores duration in 'elapsed_ms'.
    """
    timing = {}
    timing["start"] = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed = time.perf_counter() - timing["start"]
        timing["elapsed_ms"] = elapsed * 1000.0


def compute_status(score: float, *, threshold_orange: float, threshold_red: float) -> str:
    """Compute status decision from anomaly score.

    Args:
        score: Anomaly score from 0 to infinity.
        threshold_orange: Score threshold for orange decision (inclusive).
        threshold_red: Score threshold for red decision (inclusive).

    Returns:
        Status string: "green", "orange", or "red".
    """
    if score >= threshold_red:
        return "red"
    if score >= threshold_orange:
        return "orange"
    return "green"


__all__ = ["compute_status", "measure_inference_time"]
