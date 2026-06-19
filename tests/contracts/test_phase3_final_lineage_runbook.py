from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_phase3_final_lineage_runbook_covers_final_demo_evidence() -> None:
    content = (ROOT / "docs" / "phase3-final-lineage-runbook.md").read_text(encoding="utf-8")

    for expected in [
        "Docker Hub",
        "docker login",
        "IQA_IMAGE_TAG=sha-<commit>",
        "IQA_DOCKER_GID",
        "deploy/smoke-test.sh",
        "iqa_dvc_reproducibility",
        "iqa_gpu",
        "iqa-run-replay-lifecycle-cycle",
        "iqa-lineage-summary",
        "--require-mlflow-run",
        "MLflow Registry",
        "MinIO",
        "DVC",
        "PostgreSQL",
        "Sophie",
        "Marc",
        "Laurent",
    ]:
        assert expected in content


def test_docs_index_links_phase3_final_lineage_runbook() -> None:
    content = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "phase3-final-lineage-runbook.md" in content
    assert "soutenance" in content
