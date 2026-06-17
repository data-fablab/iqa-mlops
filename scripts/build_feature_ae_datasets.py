"""Build materialized Feature-AE v002/v003 candidate manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from iqa.datasets import (
    FEATURE_AE_GOOD_V002,
    FEATURE_AE_GOOD_V003,
    CastingImageSample,
    build_oracle_validated_feature_ae_dataset,
)


NATURAL_REPLAY = Path("data/metadata/casting_flux_replay_plan_natural.csv")
DRIFT_REPLAY = Path("data/metadata/casting_flux_replay_plan_drift.csv")
DEFAULT_OUTPUT_DIR = Path("data/model_datasets")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-replay", type=Path, default=NATURAL_REPLAY)
    parser.add_argument("--drift-replay", type=Path, default=DRIFT_REPLAY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "defective", "anomaly"}


def _split_pipe(value: str | None) -> list[str]:
    return [part for part in (value or "").split("|") if part]


def _indexed(parts: list[str], index: int, fallback: str = "") -> str:
    return parts[index] if index < len(parts) else fallback


def _samples_from_replay_manifest(path: Path) -> list[CastingImageSample]:
    samples: list[CastingImageSample] = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image_ids = _split_pipe(row.get("image_ids") or row.get("image_id"))
            relative_paths = _split_pipe(row.get("relative_paths") or row.get("relative_path"))
            is_defective = _truthy(row.get("is_defective") or "")
            oracle_verdict = "defective" if is_defective else "conforme"
            train_eligible = not is_defective
            quarantine_reason = "oracle_defective" if is_defective else ""
            for index, relative_path in enumerate(relative_paths):
                image_id = _indexed(image_ids, index, Path(relative_path).stem)
                samples.append(
                    CastingImageSample(
                        image_id=image_id,
                        relative_path=relative_path,
                        event_id=row.get("piece_event_id") or row.get("simulated_event_id") or "",
                        source_class=row.get("source_class") or "",
                        split_set=row.get("scenario_id") or "",
                        label=row.get("label") or "good",
                        is_defective=is_defective,
                        scenario_id=row.get("scenario_id") or "",
                        dataset_version=row.get("dataset_version") or "",
                        oracle_verdict=oracle_verdict,
                        train_eligible=train_eligible,
                        train_eligibility_source="oracle_gt",
                        quarantine_reason=quarantine_reason,
                    )
                )
    return samples


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    natural = build_oracle_validated_feature_ae_dataset(
        _samples_from_replay_manifest(args.natural_replay),
        args.output_dir / f"{FEATURE_AE_GOOD_V002}.csv",
        FEATURE_AE_GOOD_V002,
        manifest_version=f"{FEATURE_AE_GOOD_V002}_manifest_v001",
    )
    drift = build_oracle_validated_feature_ae_dataset(
        _samples_from_replay_manifest(args.drift_replay),
        args.output_dir / f"{FEATURE_AE_GOOD_V003}.csv",
        FEATURE_AE_GOOD_V003,
        manifest_version=f"{FEATURE_AE_GOOD_V003}_manifest_v001",
    )

    print(f"Wrote {natural.output_manifest} ({natural.sample_count} samples).")
    print(f"Wrote {drift.output_manifest} ({drift.sample_count} samples).")


if __name__ == "__main__":
    main()
