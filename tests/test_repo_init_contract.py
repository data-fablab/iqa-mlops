from __future__ import annotations

from pathlib import Path


ROOT = Path(".")


def test_phase_1_root_files_exist() -> None:
    for path in [
        "README.md",
        ".env.example",
        ".gitignore",
        ".python-version",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ]:
        assert (ROOT / path).is_file()


def test_phase_1_config_files_exist() -> None:
    for path in [
        "configs/paths.yaml",
        "configs/replay_scenarios.yaml",
        "configs/monitoring_thresholds.yaml",
        "configs/promotion_gates.yaml",
    ]:
        assert (ROOT / path).is_file()


def test_validation_set_contract_is_documented() -> None:
    readme = ROOT / "data" / "validation" / "README.md"

    assert readme.is_file()
    assert "validation_set_v001" in readme.read_text(encoding="utf-8")


def test_metadata_store_is_documented_as_postgresql() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    docs_and_examples = [
        ROOT / "README.md",
        ROOT / "docs" / "Configuration-Serveur-IQA.md",
        ROOT / "deploy" / "mvp_simulation" / ".env.sim.example",
        ROOT / "reports" / "mvp_environment_simulation" / "README.md",
    ]

    assert "IQA_METADATA_DB_URL=postgresql://" in env_example
    for path in docs_and_examples:
        content = path.read_text(encoding="utf-8").lower()
        assert "postgresql" in content or "postgres" in content
        assert "sqlite" not in content


def test_ingestion_abstraction_is_documented() -> None:
    expected_terms = ["historical_replay", "production_ingest", "iqa-ingested-images"]
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "Architecture-Projet-IQA.md",
        ROOT / "docs" / "Cadrage-Projet-MLOps-IQA.md",
        ROOT / "docs" / "Configuration-Serveur-IQA.md",
        ROOT / "docs" / "adr" / "0003-minio-stockage-objet-local.md",
        ROOT / "docs" / "adr" / "0004-postgresql-comme-metadata-store.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)

    for term in expected_terms:
        assert term in combined


def test_no_heavy_model_or_sqlite_artifacts_in_repo_tree() -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".onnx", ".sqlite"}
    offenders = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in forbidden_suffixes
        and ".venv" not in path.parts
    ]

    assert offenders == []
