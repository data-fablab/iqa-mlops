"""Domain exceptions for the promotion workflow."""

from __future__ import annotations


class PromotionBlockedError(RuntimeError):
    """Raised when a model cannot be promoted (gates failed or a step blocked it).

    Used by lifecycle tasks so callers (Airflow) can fail the task on a specific,
    catchable error rather than a bare ``Exception``.
    """


__all__ = ["PromotionBlockedError"]
