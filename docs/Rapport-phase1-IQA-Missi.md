# Rapport Phase 1 - IQA MLOps

Ce document recapitule les travaux realises pour la Phase 1 (squelette repo,
infra Docker Compose, CI), en complement de `Architecture-Projet-IQA.md` et
`Runbook-Phase1-IQA.md`.

## 1. Analyse de coherence architecture

Comparaison du repo avec `Architecture-Projet-IQA.md` et les contrats
`tests/test_architecture_contract.py` / `tests/test_repo_init_contract.py`.
Constat initial : squelette Phase 1 incomplet (fichiers de support `deploy/*`
manquants, test `test_deploy_support_directories_exist` en echec).

## 2. Squelette Phase 1 - fichiers de support deploy (commit `88ef03e`)

- `deploy/postgres/init-databases.sql` : provisionne les trois bases
  (`iqa_metadata`, `mlflow`, `airflow`) sous l'utilisateur `iqa`.
- `deploy/minio/init-buckets.sh` : cree les huit buckets
  (`iqa-source-datasets`, `iqa-dvc`, `iqa-ingested-images`,
  `mlflow-artifacts`, `iqa-roi-masks`, `iqa-heatmaps`, `iqa-models`,
  `iqa-backups`).
- `deploy/docker-compose.yml` minimal initial.
- Correction `.gitignore` : `postgres/`, `minio/`, `prometheus/`,
  `grafana/` -> `/postgres/`, `/minio/`, `/prometheus/`, `/grafana/` (les
  motifs non ancres masquaient `deploy/postgres/`, etc.).

## 3. Airflow GPU pool / Streamlit / CI / Runbook (commit `b5bfff3`, merge sur `main` via PR #1)

- **Airflow** : service `airflow-init` (db migrate + import pools),
  `deploy/airflow/pools.json` -> pool `iqa_gpu` (1 slot) pour contraindre
  `max_active_tasks=1` sur les taches GPU du DAG `iqa_lifecycle`.
  `airflow-webserver` / `airflow-scheduler` en `LocalExecutor`, dependent de
  `airflow-init` (`service_completed_successfully`).
- **Streamlit** : `deploy/streamlit/app.py` - vitrine placeholder (modele
  actif, lots de replay, statut piece, formulaire feedback `oracle_gt`),
  service `iqa-streamlit` dans docker-compose.
- **CI** : `.github/workflows/ci.yml` - `uv sync`, `ruff check`, `pytest -q`
  sur push/PR vers `main`.
- **Runbook** : `docs/Runbook-Phase1-IQA.md` - prerequis, install locale,
  `.env`, demarrage Docker Compose par etapes, arret, rappel livrable
  Phase 1.

## 4. Docker-compose complet / PostgreSQL / MinIO lifecycle (commit `adbb13c`)

- `deploy/docker-compose.yml` finalise : tous les services (api, inference,
  streamlit, ingestion/replay/trainer/monitoring en `profiles: batch`,
  airflow, mlflow, minio, postgres, prometheus, grafana, reverse-proxy).
- `deploy/minio/lifecycle-heatmaps.json` : regle ILM `expire-heatmap-lots` -
  expire `lots/` apres 30 jours, `curated/` non concerne.
- `deploy/minio/init-buckets.sh` : ajout `mc ilm import` pour appliquer la
  policy sur `iqa-heatmaps`.
- `minio-init` monte desormais aussi `lifecycle-heatmaps.json`.

## 5. Etat actuel

- 72/72 tests passent, working tree propre.
- Travaux sur la branche `feature/airflow-gpu-pool-streamlit-ci-runbook`.
- `b5bfff3` est deja merge sur `origin/main` (PR #1).
- `adbb13c` (lifecycle heatmaps) n'est pas encore poussee :

```bash
git push origin feature/airflow-gpu-pool-streamlit-ci-runbook
gh pr create --title "Add heatmap lifecycle policy" --body "Adds MinIO ILM rule expiring lots/ heatmaps after 30 days; curated/ kept."
```

## 6. Livrable Phase 1 (rappel `Architecture-Projet-IQA.md`)

- dataset source identifie
- contrat ingestion defini
- buckets documentes
- PostgreSQL positionne comme metadata store
- artefacts lourds exclus de Git (DVC)
- squelette Docker Compose demarrable (postgres, minio, api `/health`)
- CI minimale (lint + tests)

Phase 1 complete cote squelette/infra.

## 7. Point en cours - revue merge `Ken_branch`

Conflit textuel unique dans `src/iqa/inference/__init__.py` entre `main` et
`Ken_branch`. Sous-jacent : `Ken_branch` renomme le champ
`InferenceResult.decision` -> `statut` (+ ajoute `roi_status`,
`PieceEvent`) dans `contracts.py`, fichier non modifie par `main`. Une fusion
3-way adopterait silencieusement cette version et casserait
`src/iqa/api/main.py` (`prediction["decision"]`) et
`tests/test_api_skeleton.py` (assertions `["decision"]`). A regler avant le
merge (choisir un seul nom de champ et l'appliquer partout).
