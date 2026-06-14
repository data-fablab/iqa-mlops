# Runbook Phase 1 - Squelette IQA

Ce runbook couvre le perimetre de la Phase 1 (cf. `Architecture-Projet-IQA.md`
section 10 et `Repartition-Taches-Phases-1-2.md`) : squelette repo, configs,
CI, PostgreSQL, MinIO, API `/health`. Il decrit comment installer le projet et
demarrer le socle Docker Compose sur la station de travail / le serveur Z420.

## 1. Prerequis

- Ubuntu Server (cible officielle) ou poste de dev Linux/WSL2.
- `uv` installe (https://docs.astral.sh/uv/).
- Docker + Docker Compose plugin v2.
- Python pilote par `uv` via `.python-version` (3.12) ; pas d'installation
  manuelle requise.

## 2. Installation locale (sans Docker)

```bash
git clone <repo>
cd iqa-mlops
make sync      # uv sync --extra cpu
make lint      # ruff check src scripts tests
make test      # uv run pytest -q
```

`make test` valide les contrats : DAGs Airflow importables, services Docker
Compose presents, configs presentes, documentation a jour (voir
`tests/test_architecture_contract.py` et `tests/test_repo_init_contract.py`).

## 3. Configuration de l'environnement

```bash
cp .env.example .env
```

Adapter au minimum, dans `.env` :

- `IQA_POSTGRES_PASSWORD`
- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`
- `GF_SECURITY_ADMIN_PASSWORD`
- `IQA_ADMIN_TOKEN` / `IQA_SERVICE_TOKEN`

`.env` est ignore par Git (`.gitignore`) ; `.env.example` reste la reference
versionnee.

## 4. Demarrage du socle Docker Compose

```bash
cd deploy
docker compose up -d postgres minio minio-init
```

- `postgres` : un conteneur, trois bases (`iqa_metadata`, `mlflow`,
  `airflow`), provisionnees par `deploy/postgres/init-databases.sql`.
- `minio` + `minio-init` : objet store local, buckets crees par
  `deploy/minio/init-buckets.sh` (`iqa-source-datasets`, `iqa-dvc`,
  `iqa-ingested-images`, `mlflow-artifacts`, `iqa-roi-masks`, `iqa-heatmaps`,
  `iqa-models`, `iqa-backups`).

Verifier :

```bash
docker compose ps
docker compose logs minio-init
```

## 5. API `/health`

```bash
docker compose up -d iqa-api iqa-inference
curl http://localhost:8000/health
# {"status": "ok", "service": "iqa-api"}
```

## 6. Observabilite et reverse proxy

```bash
docker compose up -d prometheus grafana reverse-proxy
```

- Prometheus scrape `iqa-api:8000/metrics` et `iqa-inference:8100/metrics`
  (config `deploy/prometheus/prometheus.yml`).
- Grafana est accessible derriere `reverse-proxy` (`/grafana/`), provisioning
  dans `deploy/grafana/provisioning/`.

## 7. Airflow (LocalExecutor, pool GPU)

```bash
docker compose up airflow-init   # une fois : db migrate + import pools.json
docker compose up -d airflow-webserver airflow-scheduler
```

- `AIRFLOW__CORE__EXECUTOR=LocalExecutor` (pas de Celery/Kubernetes).
- Le pool `iqa_gpu` (1 slot) est importe depuis `deploy/airflow/pools.json` et
  contraint les taches GPU du DAG `iqa_lifecycle` a `max_active_tasks=1`.

## 8. Streamlit (vitrine Sophie, placeholder)

```bash
docker compose up -d iqa-streamlit
```

Ouvre `http://localhost:8501` : modele actif (`/model/version`), lots
(`/replay-scenarios`), statut piece (`/piece-events/{id}/predict`) et
formulaire feedback `oracle_gt` (`/feedback`). C'est une vitrine MVP ; aucun
historique PostgreSQL n'est encore branche en Phase 1.

## 9. CI

`.github/workflows/ci.yml` execute `ruff check` puis `pytest -q` sur chaque
push/PR vers `main`. La CI ne declenche jamais d'entrainement (decision
Phase 1, cf. `Architecture-Projet-IQA.md` section 11).

## 10. Tout arreter

```bash
docker compose down
```

Les volumes nommes (`postgres_data`, `minio_data`) sont conserves entre les
demarrages ; utiliser `docker compose down -v` pour repartir d'un etat vide.

## 11. Livrable Phase 1 (rappel)

```text
dataset source identifie
contrat ingestion defini
buckets documentes
PostgreSQL positionne comme metadata store
artefacts lourds exclus de Git
squelette Docker Compose demarrable : postgres, minio, api /health
CI minimale (lint + tests)
```
