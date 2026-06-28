"""Idempotent demo reset — restore the clean class1-only start (handoff limite #5).

The autonomous lifecycle mutates the *active* serving artifacts in place
(``covered_classes`` grows class1 -> class2 -> class3 as it promotes). After a
rehearsal the stack is therefore "dirty": streaming class2 against a PatchCore
that already covers class2 produces no drift, so the demo cannot start.

Rather than rebuild on the GPU, this restores the active artifacts from the
**immutable baselines that already exist** — a pure file copy, no GPU:

- PatchCore bank: ``patchcore_domain_drift_v001`` (pristine class1-only memory
  bank + calibration) -> ``patchcore_domain_drift_active``, with the active
  manifest's ``covered_classes`` pinned back to ``[Casting_class1]``.
- Feature-AE: ``rd_feature_ae_class1_baseline/checkpoint.pt`` -> the fixed active
  checkpoint the inference container loads.
- Drift-state band file -> ``{Casting_class1: covered}``.

Because it always copies *from the baselines*, the command is **idempotent**: run
it at runsheet Phase 0 and you get the same clean class1-only start regardless of
how dirty the previous run left things — no manual reset, no leftover state.

Run from the repo root with the stack up::

    .venv/Scripts/python.exe -m scripts.demo_reset

``--no-restart`` skips the container restart (e.g. when the stack is down).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / ".cache" / "iqa" / "models"
DEFAULT_DRIFT_STATE = REPO_ROOT / "deploy" / "drift-state" / "state.json"

# Pristine, never-mutated baselines (the lifecycle only ever writes the *_active
# dirs, so these stay class1-only).
PATCHCORE_BASELINE_DIR = "patchcore_domain_drift_v001"
PATCHCORE_ACTIVE_DIR = "patchcore_domain_drift_active"
PATCHCORE_DATA_FILES = ("memory_bank.pt", "calibration.yaml", "class_scores.csv")
PATCHCORE_MANIFEST = "model_manifest.json"

AE_BASELINE_DIR = "rd_feature_ae_class1_baseline"
AE_ACTIVE_DIR = "rd_feature_ae_active"
AE_CHECKPOINT = "checkpoint.pt"

BASELINE_CLASS = "Casting_class1"

# Containers to bounce so the freshly-restored artifacts (and the API metric code)
# are reloaded. iqa-inference reloads the models; iqa-api reloads main.py.
DEFAULT_INFERENCE_CONTAINER = "deploy-iqa-inference-1"
DEFAULT_API_CONTAINER = "deploy-iqa-api-1"


@dataclass
class ResetResult:
    """Summary of what the restore touched (for logging / tests)."""

    patchcore_files: list[str] = field(default_factory=list)
    covered_classes: list[str] = field(default_factory=list)
    ae_checkpoint: str | None = None
    drift_state: str | None = None
    restarted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "patchcore_files": self.patchcore_files,
            "covered_classes": self.covered_classes,
            "ae_checkpoint": self.ae_checkpoint,
            "drift_state": self.drift_state,
            "restarted": self.restarted,
        }


def restore_class1_baseline(
    models_dir: Path, drift_state_path: Path
) -> ResetResult:
    """Restore active PatchCore + AE + drift-state to the class1-only baseline.

    Pure filesystem work (no Docker), so it is unit-testable and idempotent: it
    only ever reads from the baselines and overwrites the active artifacts.
    """
    result = ResetResult()

    pc_baseline = models_dir / PATCHCORE_BASELINE_DIR
    pc_active = models_dir / PATCHCORE_ACTIVE_DIR
    if not pc_baseline.is_dir():
        raise FileNotFoundError(f"PatchCore baseline missing: {pc_baseline}")
    pc_active.mkdir(parents=True, exist_ok=True)

    for name in PATCHCORE_DATA_FILES:
        src = pc_baseline / name
        if src.exists():
            shutil.copy2(src, pc_active / name)
            result.patchcore_files.append(name)

    # Active manifest = baseline manifest with covered_classes pinned to class1.
    # The detector reads covered_classes from here (domain_drift.py); pinning it
    # explicitly avoids depending on the default and survives a BOM-tolerant read.
    baseline_manifest_path = pc_baseline / PATCHCORE_MANIFEST
    manifest: dict = {}
    if baseline_manifest_path.exists():
        manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8-sig"))
    manifest["covered_classes"] = [BASELINE_CLASS]
    (pc_active / PATCHCORE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result.covered_classes = [BASELINE_CLASS]
    result.patchcore_files.append(PATCHCORE_MANIFEST)

    # Feature-AE: restore the fixed active checkpoint the inference container loads.
    ae_src = models_dir / AE_BASELINE_DIR / AE_CHECKPOINT
    ae_active_dir = models_dir / AE_ACTIVE_DIR
    if not ae_src.exists():
        raise FileNotFoundError(f"AE baseline checkpoint missing: {ae_src}")
    ae_active_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ae_src, ae_active_dir / AE_CHECKPOINT)
    result.ae_checkpoint = str(ae_active_dir / AE_CHECKPOINT)

    # Drift-state band file (deterministic scorer / Grafana coverage view).
    drift_state_path.parent.mkdir(parents=True, exist_ok=True)
    drift_state_path.write_text(
        json.dumps({"classes": {BASELINE_CLASS: "covered"}}, indent=2) + "\n",
        encoding="utf-8",
    )
    result.drift_state = str(drift_state_path)

    return result


def _restart_containers(names: list[str]) -> list[str]:
    """``docker restart`` each container; returns the ones that restarted OK."""
    restarted: list[str] = []
    for name in names:
        proc = subprocess.run(
            ["docker", "restart", name],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            restarted.append(name)
            print(f"[demo_reset] restarted {name}")
        else:
            print(f"[demo_reset] WARN could not restart {name}: {proc.stderr.strip()}")
    return restarted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--drift-state", type=Path, default=DEFAULT_DRIFT_STATE)
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="skip the docker restart (use when the stack is down)",
    )
    parser.add_argument(
        "--inference-container",
        default=os.environ.get("IQA_INFERENCE_CONTAINER", DEFAULT_INFERENCE_CONTAINER),
    )
    parser.add_argument(
        "--api-container",
        default=os.environ.get("IQA_API_CONTAINER", DEFAULT_API_CONTAINER),
    )
    args = parser.parse_args(argv)

    result = restore_class1_baseline(args.models_dir, args.drift_state)
    print(f"[demo_reset] PatchCore active restored -> covered_classes={result.covered_classes}")
    print(f"[demo_reset] Feature-AE active checkpoint -> {result.ae_checkpoint}")
    print(f"[demo_reset] drift-state -> {result.drift_state}")

    if not args.no_restart:
        result.restarted = _restart_containers(
            [args.api_container, args.inference_container]
        )
    else:
        print("[demo_reset] --no-restart: containers not bounced (restart them to reload artifacts)")

    print("[demo_reset] clean class1-only start ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
