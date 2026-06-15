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


def test_convergence_adrs_exist() -> None:
    for path in [
        ROOT / "docs" / "adr" / "0005-calibration-set-etanche-et-split-piece-event.md",
        ROOT / "docs" / "adr" / "0006-mlflow-registry-source-verite.md",
        ROOT / "docs" / "adr" / "0007-architecture-services-avec-pyproject-racine.md",
    ]:
        assert path.is_file()


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
    assert "IQA_MLFLOW_DB_URL=postgresql://" in env_example
    assert "IQA_AIRFLOW_DB_URL=postgresql://" in env_example
    for path in docs_and_examples:
        content = path.read_text(encoding="utf-8").lower()
        assert "postgresql" in content or "postgres" in content
        assert "sqlite" not in content


def test_architecture_documents_microservices_and_registry_truth() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "docs" / "Architecture-Projet-IQA.md",
            ROOT / "docs" / "Configuration-Serveur-IQA.md",
            ROOT / "docs" / "Decisions-Questions-Ouvertes-IQA.md",
        ]
    )

    for service in ["iqa-api", "iqa-inference", "iqa-ingestion", "iqa-replay", "iqa-trainer", "iqa-monitoring"]:
        assert service in docs
    assert "MLflow Registry est la source de verite" in docs
    assert "Reverse proxy | Nginx" in docs
    assert "Nginx/Traefik" not in docs


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


def test_convergence_decisions_are_documented() -> None:
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            ROOT / "README.md",
            ROOT / "docs" / "Architecture-Projet-IQA.md",
            ROOT / "docs" / "Cadrage-Projet-MLOps-IQA.md",
            ROOT / "docs" / "PRD-IQA-MVP.md",
            ROOT / "docs" / "Configuration-Serveur-IQA.md",
            ROOT / "docs" / "adr" / "0005-calibration-set-etanche-et-split-piece-event.md",
            ROOT / "docs" / "adr" / "0006-mlflow-registry-source-verite.md",
            ROOT / "docs" / "adr" / "0007-architecture-services-avec-pyproject-racine.md",
        ]
    )

    for term in [
        "calibration_set_v001",
        "event_time",
        "recorded_at",
        "MLflow Registry",
        "iqa-source-datasets",
        "iqa-ingested-images",
        "pyproject.toml racine",
        "feature_ae__production_replay_natural",
        "feature_ae__drift_domain_extension",
    ]:
        assert term in docs


def test_no_heavy_model_or_sqlite_artifacts_in_repo_tree() -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".onnx", ".sqlite"}
    # Exclude: venv, cache, MLflow artifacts (temp test outputs), pytest temp files
    excluded_dirs = {".venv", ".pytest_cache", "mlruns", "__pycache__", ".mypy_cache"}
    offenders = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in forbidden_suffixes
        and not any(excluded in path.parts for excluded in excluded_dirs)
    ]

    assert offenders == []
