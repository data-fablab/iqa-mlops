from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


ROOT = Path(".")


@lru_cache(maxsize=1)
def _docs_corpus() -> str:
    """All Markdown docs + the root README concatenated into one searchable blob.

    Term-presence contracts only care that a concept is documented *somewhere*, so
    we scan the docs tree instead of hardcoding file paths. Renaming or moving a
    doc no longer breaks these tests as long as the content survives.
    """
    sources = sorted((ROOT / "docs").rglob("*.md"))
    sources.append(ROOT / "README.md")
    return "\n".join(p.read_text(encoding="utf-8") for p in sources if p.is_file())


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
    # Match by ADR number, not the descriptive slug, so renaming the slug does not
    # break the contract that these decisions are recorded.
    adr_dir = ROOT / "docs" / "adr"
    for number in ("0005", "0006", "0007"):
        assert list(adr_dir.glob(f"{number}-*.md")), f"ADR {number} is missing"


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
        ROOT / "docs" / "configuration-serveur-iqa.md",
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


def test_readme_describes_current_phase_2_surfaces() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "API Skeleton" not in content
    assert "Phase 1 foundation" not in content
    for term in [
        "iqa_dvc_reproducibility",
        "iqa-check-dvc-reproducibility",
        "iqa-init-metadata-db",
        "iqa-demo-phase2",
        "production_replay_natural",
        "drift_domain_extension",
        "PostgreSQL",
        "MLflow",
        "MinIO",
        "Airflow",
    ]:
        assert term in content


def test_architecture_documents_microservices_and_registry_truth() -> None:
    docs = _docs_corpus()

    for service in ["iqa-api", "iqa-inference", "iqa-ingestion", "iqa-replay", "iqa-trainer", "iqa-monitoring"]:
        assert service in docs
    assert "MLflow Registry est la source de verite" in docs
    assert "Reverse proxy | Nginx" in docs
    assert "Nginx/Traefik" not in docs


def test_ingestion_abstraction_is_documented() -> None:
    docs = _docs_corpus()

    for term in ["historical_replay", "production_ingest", "iqa-ingested-images"]:
        assert term in docs


def test_convergence_decisions_are_documented() -> None:
    docs = _docs_corpus()

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
    tracked_files = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    offenders = [
        Path(path)
        for path in tracked_files
        if Path(path).suffix.lower() in forbidden_suffixes
    ]

    assert offenders == []
