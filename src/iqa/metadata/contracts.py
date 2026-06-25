"""Stable metadata contracts for IQA CSV manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only; pandas is a ``data`` role dep.
    import pandas as pd


RAW_DATASET_ID = "hss_iad_casting_raw_v1"

PHASE2_METADATA_COLUMNS = (
    "raw_dataset_id",
    "manifest_id",
    "piece_event_id",
    "dataset_version",
    "replay_id",
    "validation_id",
    "scenario_version",
)

ManifestKind = Literal["source", "bootstrap", "validation", "calibration", "replay"]
MetadataValue = str | Callable[["pd.DataFrame"], "pd.Series"]


@dataclass(frozen=True)
class MetadataManifestContract:
    key: str
    path: Path
    kind: ManifestKind
    manifest_id: str
    dataset_version: str
    historical_columns: tuple[str, ...]
    identity_column: str
    replay_id: str = ""
    validation_id: str = ""
    scenario_version: str = ""
    scenario_id: str | None = None

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (*self.historical_columns, *PHASE2_METADATA_COLUMNS)


SOURCE_EVENT_COLUMNS = (
    "event_id",
    "event_key",
    "source_class",
    "group_key",
    "source_timestamp",
    "source_date",
    "source_time",
    "split_set",
    "label",
    "is_defective",
    "n_images",
    "source_classes",
    "view_pairs",
    "has_mask",
    "image_ids",
    "relative_paths",
)

BOOTSTRAP_COLUMNS = (
    *SOURCE_EVENT_COLUMNS,
    "bootstrap_dataset_version",
    "bootstrap_sequence",
    "bootstrap_role",
)

VALIDATION_COLUMNS = (
    *SOURCE_EVENT_COLUMNS,
    "validation_set_id",
    "validation_role",
)

CALIBRATION_COLUMNS = (
    *SOURCE_EVENT_COLUMNS,
    "calibration_set_id",
    "calibration_role",
)

REPLAY_COLUMNS = (
    "simulated_event_id",
    "scenario_id",
    "scenario_type",
    "scenario_phase",
    "is_representative",
    "sequence_number",
    "scheduled_at",
    "production_date",
    "shift_id",
    "station_id",
    "lot_id",
    "sequence_in_lot",
    "piece_serial",
    "source_event_id",
    "source_event_key",
    "source_class",
    "source_group_key",
    "source_timestamp",
    "label",
    "is_defective",
    "expected_review_required",
    "n_images",
    "source_classes",
    "view_pairs",
    "image_ids",
    "relative_paths",
    "event_time",
    "recorded_at",
    "is_simulated",
    "roi_model_version",
    "feature_ae_version",
)

MANIFEST_CONTRACTS: dict[str, MetadataManifestContract] = {
    "source_events": MetadataManifestContract(
        key="source_events",
        path=Path("data/metadata/casting_piece_events.csv"),
        kind="source",
        manifest_id="casting_piece_events_v001",
        dataset_version=RAW_DATASET_ID,
        historical_columns=SOURCE_EVENT_COLUMNS,
        identity_column="event_id",
    ),
    "bootstrap_events": MetadataManifestContract(
        key="bootstrap_events",
        path=Path("data/metadata/feature_ae_bootstrap_events.csv"),
        kind="bootstrap",
        manifest_id="feature_ae_bootstrap_events_v001",
        dataset_version="feature_ae_good_v001_bootstrap",
        historical_columns=BOOTSTRAP_COLUMNS,
        identity_column="event_id",
    ),
    "natural_replay": MetadataManifestContract(
        key="natural_replay",
        path=Path("data/metadata/casting_flux_replay_plan_natural_v003.csv"),
        kind="replay",
        manifest_id="casting_flux_replay_plan_natural_v002",
        dataset_version="production_replay_natural_v002",
        historical_columns=REPLAY_COLUMNS,
        identity_column="simulated_event_id",
        replay_id="production_replay_natural_v002",
        scenario_version="casting_flux_replay_plan_natural_v003",
        scenario_id="production_replay_natural",
    ),
    "natural_replay_train_v4": MetadataManifestContract(
        key="natural_replay_train_v4",
        path=Path("data/metadata/casting_flux_replay_plan_natural_train_v004.csv"),
        kind="replay",
        manifest_id="casting_flux_replay_plan_natural_train_v004",
        dataset_version="production_replay_natural_train_v004",
        historical_columns=REPLAY_COLUMNS,
        identity_column="simulated_event_id",
        replay_id="production_replay_natural_train_v004",
        scenario_version="casting_flux_replay_plan_natural_train_v004",
        scenario_id="production_replay_natural_train_v004",
    ),
    "drift_replay": MetadataManifestContract(
        key="drift_replay",
        path=Path("data/metadata/casting_flux_replay_plan_drift.csv"),
        kind="replay",
        manifest_id="casting_flux_replay_plan_drift_v001",
        dataset_version="drift_domain_extension_v001",
        historical_columns=REPLAY_COLUMNS,
        identity_column="simulated_event_id",
        replay_id="drift_domain_extension_v001",
        scenario_version="drift_domain_extension_v001",
        scenario_id="drift_domain_extension",
    ),
    "validation_set": MetadataManifestContract(
        key="validation_set",
        path=Path("data/validation/validation_set_replay_representative_v001.csv"),
        kind="validation",
        manifest_id="validation_set_replay_representative_v001",
        dataset_version="validation_set_replay_representative_v001",
        historical_columns=VALIDATION_COLUMNS,
        identity_column="event_id",
        validation_id="validation_set_replay_representative_v001",
        scenario_version="feature_ae_mvp_v001",
    ),
    "validation_gate_set": MetadataManifestContract(
        key="validation_gate_set",
        path=Path("data/validation/validation_set_replay_gate_v001.csv"),
        kind="validation",
        manifest_id="validation_set_replay_gate_v001",
        dataset_version="validation_set_replay_gate_v001",
        historical_columns=VALIDATION_COLUMNS,
        identity_column="event_id",
        validation_id="validation_set_replay_gate_v001",
        scenario_version="feature_ae_mvp_v001",
    ),
    "validation_gate_set_v2": MetadataManifestContract(
        key="validation_gate_set_v2",
        path=Path("data/validation/validation_set_replay_gate_v002.csv"),
        kind="validation",
        manifest_id="validation_set_replay_gate_v002",
        dataset_version="validation_set_replay_gate_v002",
        historical_columns=VALIDATION_COLUMNS,
        identity_column="event_id",
        validation_id="validation_set_replay_gate_v002",
        scenario_version="feature_ae_mvp_v002",
    ),
    "validation_gate_set_v3": MetadataManifestContract(
        key="validation_gate_set_v3",
        path=Path("data/validation/validation_set_replay_gate_v003.csv"),
        kind="validation",
        manifest_id="validation_set_replay_gate_v003",
        dataset_version="validation_set_replay_gate_v003",
        historical_columns=VALIDATION_COLUMNS,
        identity_column="event_id",
        validation_id="validation_set_replay_gate_v003",
        scenario_version="feature_ae_mvp_v003",
    ),
    "calibration_set": MetadataManifestContract(
        key="calibration_set",
        path=Path("data/validation/calibration_good_reference_v001.csv"),
        kind="calibration",
        manifest_id="calibration_good_reference_v001",
        dataset_version="calibration_good_reference_v001",
        historical_columns=VALIDATION_COLUMNS,
        identity_column="event_id",
        validation_id="calibration_good_reference_v001",
        scenario_version="feature_ae_mvp_v001",
    ),
}


def metadata_columns_for(contract: MetadataManifestContract) -> dict[str, MetadataValue]:
    """Return deterministic Phase 2 metadata values for a manifest frame."""

    return {
        "raw_dataset_id": RAW_DATASET_ID,
        "manifest_id": contract.manifest_id,
        "piece_event_id": lambda frame: frame[contract.identity_column].astype(str),
        "dataset_version": contract.dataset_version,
        "replay_id": contract.replay_id,
        "validation_id": contract.validation_id,
        "scenario_version": contract.scenario_version,
    }


def apply_metadata_contract(frame: pd.DataFrame, contract: MetadataManifestContract) -> pd.DataFrame:
    """Append stable Phase 2 metadata columns without duplicating prior runs."""

    missing_historical = [column for column in contract.historical_columns if column not in frame.columns]
    if missing_historical:
        joined = ", ".join(missing_historical)
        raise ValueError(f"{contract.key} is missing historical columns: {joined}")

    normalized = frame.drop(columns=[c for c in PHASE2_METADATA_COLUMNS if c in frame.columns]).copy()
    metadata_values = metadata_columns_for(contract)
    for column in PHASE2_METADATA_COLUMNS:
        value = metadata_values[column]
        normalized[column] = value(normalized) if callable(value) else value

    return normalized


def contract_for_key(key: str) -> MetadataManifestContract:
    return MANIFEST_CONTRACTS[key]


__all__ = [
    "MANIFEST_CONTRACTS",
    "PHASE2_METADATA_COLUMNS",
    "RAW_DATASET_ID",
    "MetadataManifestContract",
    "apply_metadata_contract",
    "contract_for_key",
]
