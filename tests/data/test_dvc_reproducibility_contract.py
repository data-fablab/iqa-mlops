from __future__ import annotations

from pathlib import Path


DVC_YAML = Path("dvc.yaml")
DVC_VERSIONING_DOC = Path("docs/dvc-versioning.md")


def test_dvc_yaml_declares_phase2_data_stages() -> None:
    content = DVC_YAML.read_text(encoding="utf-8")

    for stage in ["inventory", "piece_events", "replay", "validation", "model_dataset"]:
        assert f"  {stage}:" in content

    for script in [
        "scripts/build_inventory.py",
        "scripts/finalize_data_phase1.py",
        "scripts/build_flux_plan.py",
    ]:
        assert script in content

    assert "tests/datasets/test_candidate_builder.py" in content
    assert ".pt" not in content


def test_dvc_versioning_doc_links_remote_and_contracts() -> None:
    content = DVC_VERSIONING_DOC.read_text(encoding="utf-8")

    assert "iqa-minio" in content
    assert "dvc pull" in content
    assert "dvc repro" in content
    assert "docs/data-contracts.md" in content
