from __future__ import annotations

from pathlib import Path


DVC_YAML = Path("dvc.yaml")
DVC_VERSIONING_DOC = Path("docs/dvc-versioning.md")
DVC_REPRO_SCRIPT = Path("scripts/check_dvc_reproducibility.py")
PYPROJECT = Path("pyproject.toml")


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
    assert "uv run --extra cpu --extra data dvc pull" in content
    assert "uv run --extra cpu --extra data dvc repro" in content
    assert "iqa-check-dvc-reproducibility --with-network" in content
    assert "iqa_dvc_reproducibility" in content
    assert '"with_network": true' in content
    assert "DVC est un gate de reproductibilite" in content
    assert "docs/data-contracts.md" in content


def test_dvc_reproducibility_script_checks_minio_and_manifests() -> None:
    content = DVC_REPRO_SCRIPT.read_text(encoding="utf-8")
    pyproject = PYPROJECT.read_text(encoding="utf-8")

    assert 'EXPECTED_REMOTE_NAME = "iqa-minio"' in content
    assert 'EXPECTED_REMOTE_URL = "s3://iqa-dvc"' in content
    assert '["dvc", "pull", dvc_target]' in content
    assert '["dvc", "push", dvc_target]' in content
    assert "scripts/finalize_data_phase1.py" in content
    assert "git\", \"diff\", \"--quiet\"" in content
    assert "iqa-check-dvc-reproducibility" in pyproject
