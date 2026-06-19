import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "phase3_lineage_evidence.ipynb"
README = ROOT / "notebooks" / "README.md"


def _notebook_text() -> str:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])


def test_phase3_lineage_notebook_covers_soutenance_evidence() -> None:
    content = _notebook_text()

    for expected in [
        "iqa-lineage-summary",
        "iqa-run-replay-lifecycle-cycle",
        "--require-mlflow-run",
        "MLflow Registry",
        "MinIO",
        "DVC",
        "PostgreSQL",
        "Docker Hub",
        "iqa_dvc_reproducibility",
        "iqa_gpu",
        "Sophie",
        "Marc",
        "Laurent",
    ]:
        assert expected in content


def test_notebook_readme_documents_server_jupyter_path() -> None:
    content = README.read_text(encoding="utf-8")

    for expected in [
        "ssh -L 8888:localhost:8888",
        "jupyter lab",
        "uv run --with jupyter --with ipykernel",
        "VS Code Remote SSH",
        "phase3_lineage_evidence.ipynb",
    ]:
        assert expected in content


def test_docs_index_references_phase3_lineage_notebook() -> None:
    content = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")

    assert "notebooks/README.md" in content
    assert "phase3_lineage_evidence.ipynb" in content
