"""Refresh active serving artifacts after a promotion (Issue 23).

Reads the cycle ``summary.json``. When ``promotion_status=promoted``:

1. Copies the champion checkpoint to a **fixed** ``rd_feature_ae_active/`` path.
2. Rebuilds the PatchCore bank to cover class1 + the newly-covered class
   (union via the active manifest, balanced build + union calibration).

The "fixed active paths" design means the inference container always loads the
same mount point; we overwrite in place and restart the container (Issue 24).

When there is no promotion the task is a no-op (clean skip).
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from iqa.inference.domain_drift import (
    DEFAULT_DETECTOR_DIR,
    PatchCoreDomainDriftDetector,
    union_covered_classes,
)

logger = logging.getLogger(__name__)

ACTIVE_FEATURE_AE_DIR = "rd_feature_ae_active"
ACTIVE_DETECTOR_DIR = "patchcore_domain_drift_active"
ACTIVE_CHECKPOINT_NAME = "checkpoint.pt"


@dataclass
class RefreshResult:
    status: str
    feature_ae_checkpoint: str | None = None
    detector_dir: str | None = None
    covered_classes: list[str] | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "feature_ae_checkpoint": self.feature_ae_checkpoint,
            "detector_dir": self.detector_dir,
            "covered_classes": self.covered_classes,
            "reason": self.reason,
        }


def _is_promoted(summary: dict) -> bool:
    if summary.get("promotion_status") == "promoted":
        return True
    for cycle in summary.get("comparison_history", []):
        if cycle.get("promotion_status") == "promoted":
            return True
    models_promoted = summary.get("models_promoted", [])
    return bool(models_promoted)


def _champion_checkpoint(summary: dict) -> str | None:
    path = summary.get("candidate_checkpoint")
    if path:
        return str(path)
    models = summary.get("models_promoted", [])
    if models:
        last = models[-1]
        candidate_path = summary.get("candidate_checkpoint") or ""
        if candidate_path:
            return candidate_path
        base = Path(".cache/iqa/models") / last / "checkpoint.pt"
        return str(base)
    return None


def _triggering_class(summary: dict) -> str | None:
    for cycle in reversed(summary.get("comparison_history", [])):
        if cycle.get("promotion_status") == "promoted":
            version = cycle.get("candidate_version", "")
            if "drift" in version or "class2" in version.lower():
                return "Casting_class2"
            if "class3" in version.lower():
                return "Casting_class3"
    trigger_reason = summary.get("trigger_reason", "")
    if "drift" in trigger_reason:
        return summary.get("triggering_class") or "Casting_class2"
    return None


def resolve_new_covered_classes(
    detector_dir: str | Path,
    triggering_class: str | None,
) -> list[str]:
    """Derive the new covered set from the active detector manifest + triggering class."""
    manifest_path = Path(detector_dir) / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = manifest.get("covered_classes", ["Casting_class1"])
    else:
        current = ["Casting_class1"]

    if triggering_class:
        return union_covered_classes(current, triggering_class)
    return sorted(set(current))


def refresh_active_artifacts(
    summary: dict,
    models_root: str | Path,
    *,
    build_bank_fn=None,
) -> RefreshResult:
    """Copy champion AE + rebuild PatchCore bank for the new coverage.

    ``build_bank_fn`` is injectable for testing (avoids GPU in unit tests).
    When ``None``, the real PatchCore rebuild is invoked.
    """
    if not _is_promoted(summary):
        logger.info("no promotion in this cycle; skipping artifact refresh")
        return RefreshResult(status="skipped", reason="no_promotion")

    root = Path(models_root)
    champion = _champion_checkpoint(summary)
    if not champion:
        return RefreshResult(status="error", reason="champion_checkpoint_not_found")

    # 1. Copy champion checkpoint
    active_ae = root / ACTIVE_FEATURE_AE_DIR
    active_ae.mkdir(parents=True, exist_ok=True)
    dest_ckpt = active_ae / ACTIVE_CHECKPOINT_NAME
    shutil.copy2(champion, dest_ckpt)
    logger.info("champion checkpoint copied to %s", dest_ckpt)

    # 2. Rebuild PatchCore bank with extended coverage
    active_det = root / ACTIVE_DETECTOR_DIR
    triggering = _triggering_class(summary)
    new_covered = resolve_new_covered_classes(active_det, triggering)
    logger.info("new covered classes: %s", new_covered)

    if build_bank_fn is not None:
        build_bank_fn(
            output_dir=str(active_det),
            covered_classes=new_covered,
        )
    else:
        _default_build_bank(active_det, new_covered)

    return RefreshResult(
        status="refreshed",
        feature_ae_checkpoint=str(dest_ckpt),
        detector_dir=str(active_det),
        covered_classes=new_covered,
    )


def _default_build_bank(output_dir: Path, covered_classes: list[str]) -> None:
    """Invoke the PatchCore build script for real GPU rebuilds."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "scripts.build_patchcore_domain_drift",
        "--output-dir", str(output_dir),
        "--cover-classes", *covered_classes,
        "--no-mlflow",
    ]
    logger.info("rebuilding PatchCore bank: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def task_refresh_active_artifacts(**context) -> dict:
    """Airflow task entry point for refresh_active_artifacts."""
    params = context.get("params", {})
    summary_path = params.get("summary_path")
    if not summary_path:
        return {"status": "skipped", "reason": "no_summary_path"}

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    models_root = params.get("models_root", ".cache/iqa/models")

    result = refresh_active_artifacts(summary, models_root)
    return result.to_dict()
