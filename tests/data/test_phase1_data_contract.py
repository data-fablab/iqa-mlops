from __future__ import annotations

import configparser
import csv
from pathlib import Path


ROOT = Path(".")
METADATA = ROOT / "data" / "metadata"
VALIDATION = ROOT / "data" / "validation"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _ids(rows: list[dict[str, str]], key: str = "event_id") -> set[str]:
    return {row[key] for row in rows}


def _counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row[key]
        counts[value] = counts.get(value, 0) + 1
    return counts


def test_phase1_core_manifest_volumes_are_stable() -> None:
    piece_events = _read_csv(METADATA / "casting_piece_events.csv")
    bootstrap = _read_csv(METADATA / "feature_ae_bootstrap_events.csv")

    assert len(piece_events) == 962
    assert sum(row["is_defective"].lower() == "true" for row in piece_events) == 40
    assert sum(int(row["n_images"]) for row in piece_events) == 2183
    assert len(bootstrap) == 50
    assert {row["label"] for row in bootstrap} == {"good"}
    assert {row["is_defective"].lower() for row in bootstrap} == {"false"}


def test_casting_image_inventory_is_complete_and_linked_to_piece_events() -> None:
    inventory = _read_csv(METADATA / "casting_images_inventory.csv")
    piece_events = _read_csv(METADATA / "casting_piece_events.csv")
    event_ids = _ids(piece_events)
    image_ids = [row["image_id"] for row in inventory]

    assert len(inventory) == 2183
    assert len(image_ids) == len(set(image_ids))
    assert {row["event_id"] for row in inventory} <= event_ids
    assert all(row["relative_path"] for row in inventory)
    assert all(row["sha256"] for row in inventory)
    assert all(row["source_path_exists"] == "true" for row in inventory)
    assert all(row["has_gt_mask"] == "true" for row in inventory if row["is_defective"] == "true")


def test_validation_calibration_bootstrap_and_replay_are_disjoint() -> None:
    bootstrap_ids = _ids(_read_csv(METADATA / "feature_ae_bootstrap_events.csv"))
    validation_ids = _ids(_read_csv(VALIDATION / "validation_set_replay_representative_v001.csv"))
    calibration_ids = _ids(_read_csv(VALIDATION / "calibration_good_reference_v001.csv"))
    natural_replay_ids = _ids(_read_csv(METADATA / "casting_flux_replay_plan_natural_v003.csv"), "source_event_id")
    replay_ids = natural_replay_ids

    assert bootstrap_ids.isdisjoint(validation_ids)
    assert bootstrap_ids.isdisjoint(calibration_ids)
    assert bootstrap_ids.isdisjoint(replay_ids)
    assert validation_ids.isdisjoint(calibration_ids)
    assert validation_ids.isdisjoint(replay_ids)
    assert calibration_ids.isdisjoint(replay_ids)


def test_validation_and_calibration_roles_are_explicit() -> None:
    validation = _read_csv(VALIDATION / "validation_set_replay_representative_v001.csv")
    calibration = _read_csv(VALIDATION / "calibration_good_reference_v001.csv")

    assert {row["validation_set_id"] for row in validation} == {"validation_set_replay_representative_v001"}
    assert len(validation) == 74
    assert _counts(validation, "source_class") == {
        "Casting_class1": 13,
        "Casting_class2": 39,
        "Casting_class3": 22,
    }
    assert any(row["is_defective"].lower() == "true" for row in validation)
    assert {row["validation_set_id"] for row in calibration} == {"calibration_good_reference_v001"}
    assert {row["label"] for row in calibration} == {"good"}
    assert {row["is_defective"].lower() for row in calibration} == {"false"}


def test_replay_plans_carry_phase1_runtime_metadata() -> None:
    for path, scenario_id, dataset_version in [
        (METADATA / "casting_flux_replay_plan_natural_v003.csv", "production_replay_natural", "production_replay_natural_v002"),
        (METADATA / "casting_flux_replay_plan_drift.csv", "drift_domain_extension", "drift_domain_extension_v001"),
    ]:
        rows = _read_csv(path)
        assert rows
        assert {row["scenario_id"] for row in rows} == {scenario_id}
        assert {row["dataset_version"] for row in rows} == {dataset_version}
        assert all(row["event_time"] for row in rows)
        assert all(row["recorded_at"] for row in rows)
        assert {row["is_simulated"].lower() for row in rows} == {"true"}
        assert {row["roi_model_version"] for row in rows} == {"roi_segmenter_v001_fixed"}
        assert {row["feature_ae_version"] for row in rows} == {"rd_feature_ae_gated_v001_bootstrap"}


def test_dvc_remote_is_configured_for_phase1_data() -> None:
    config_path = ROOT / ".dvc" / "config"
    assert config_path.is_file()

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    assert config["core"]["remote"] == "iqa-minio"
    remote_section = next(section for section in config.sections() if "iqa-minio" in section)
    assert config[remote_section]["url"] == "s3://iqa-dvc"
    assert config[remote_section]["endpointurl"] == "http://localhost:9000"


def test_data_phase1_validation_report_exists() -> None:
    report = ROOT / "reports" / "data_phase1_validation.md"

    assert report.is_file()
    content = report.read_text(encoding="utf-8")
    assert "piece_events: 962" in content
    assert "bootstrap ∩ calibration ∩ replay ∩ validation = empty" in content
