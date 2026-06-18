"""Check DVC remote wiring and deterministic metadata regeneration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DVC_SOURCE_TARGET = "data/raw/hss-iad.dvc"
EXPECTED_REMOTE_NAME = "iqa-minio"
EXPECTED_REMOTE_URL = "s3://iqa-dvc"
REGENERATED_MANIFESTS = [
    Path("data/metadata/casting_piece_events.csv"),
    Path("data/metadata/feature_ae_bootstrap_events.csv"),
    Path("data/metadata/casting_flux_replay_plan_natural.csv"),
    Path("data/metadata/casting_flux_replay_plan_drift.csv"),
    Path("data/metadata/calibration_set_v001.csv"),
    Path("data/validation/validation_set_v001.csv"),
    Path("reports/data_phase1_validation.md"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-network",
        action="store_true",
        help="Run DVC pull/push against the configured MinIO remote.",
    )
    parser.add_argument(
        "--skip-regeneration",
        action="store_true",
        help="Skip deterministic metadata regeneration check.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _check_dvc_remote() -> None:
    result = _run(["dvc", "remote", "list"])
    if EXPECTED_REMOTE_NAME not in result.stdout or EXPECTED_REMOTE_URL not in result.stdout:
        raise SystemExit(
            f"Expected DVC remote {EXPECTED_REMOTE_NAME} -> {EXPECTED_REMOTE_URL}, got:\n{result.stdout}"
        )


def _check_dvc_network() -> None:
    if not Path(DVC_SOURCE_TARGET).exists():
        raise SystemExit(f"Missing DVC source target: {DVC_SOURCE_TARGET}")
    _run(["dvc", "pull", DVC_SOURCE_TARGET])
    _run(["dvc", "push", DVC_SOURCE_TARGET])


def _check_regeneration_is_clean() -> None:
    _run([sys.executable, "scripts/finalize_data_phase1.py"])
    paths = [str(path) for path in REGENERATED_MANIFESTS if path.exists()]
    diff = subprocess.run(["git", "diff", "--quiet", "--", *paths], check=False)
    if diff.returncode != 0:
        raise SystemExit("Metadata regeneration produced a Git diff. Run git diff for details.")


def main() -> None:
    args = parse_args()
    _check_dvc_remote()
    if args.with_network:
        _check_dvc_network()
    if not args.skip_regeneration:
        _check_regeneration_is_clean()
    print("DVC remote and metadata reproducibility checks passed.")


if __name__ == "__main__":
    main()
