# IQA Notebooks

These notebooks are soutenance and handover supports. They summarize the
evidence produced by the executable runbooks, but they do not replace the
operational procedures in `docs/`.

## Phase 3 Lineage Notebook

- `phase3_lineage_evidence.ipynb` explains the Phase 3 lineage chain:
  Docker Hub deployment, Airflow container runtime, DVC/MinIO data lineage,
  model checkpoints in MinIO, MLflow Registry governance, PostgreSQL metadata
  boundaries and the Sophie / Marc / Laurent demo reading.
- The notebook is readable without execution.
- Optional cells can load an existing replay lifecycle `summary.json` if
  `RUN_DIR` points to a local run directory.

## Phase 2 to Phase 3 Transition Notebook

* `phase2_to_phase3_transition.ipynb` summarizes the validated Phase 2 outcomes and the transition to Phase 3.
* It documents governance, traceability, PostgreSQL validation, Kong runtime evidence, and the Phase 3 target architecture.
* Its Python cells perform read only repository checks and do not start services or modify project files.
* The notebook was validated on CubeAI and on the IQA server at commit `2c1541e`.
* Five code cells executed successfully with zero errors and no sensitive output.
* The notebook is intended for the final presentation, mentor review, and project handover.

## Open the Phase 2 to Phase 3 Notebook on CubeAI

Recommended with VS Code Remote SSH:

```text
/mnt/nvme/mlops/iqa-mlops/notebooks/phase2_to_phase3_transition.ipynb
```

Select the repository Python environment as the notebook kernel.

## Run the Phase 3 Lineage Notebook on the Server

Recommended from your local machine:

```powershell
ssh -L 8888:localhost:8888 iqa@iqa-serveur
```

Then on the server:

```bash
cd /opt/iqa/iqa-mlops
git checkout main
git pull --ff-only

set -a
source .env
set +a
export IQA_S3_ENDPOINT_URL=http://localhost:9000

uv run --with jupyter --with ipykernel \
  jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

Open the Jupyter URL with token from your local browser:

```text
http://localhost:8888
```

Alternative: use VS Code Remote SSH, open
`/opt/iqa/iqa-mlops/notebooks/phase3_lineage_evidence.ipynb`, and select the
repository Python environment as kernel.

