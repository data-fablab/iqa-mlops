# Runbook Phase 1 - Squelette IQA

Ce runbook couvre le perimetre de la Phase 1 (cf. `architecture-iqa.md`
section 12 et `repartition-taches-phases-1-2.md`) : squelette repo, configs,
CI, PostgreSQL infra, MinIO, API `/health`, ROI bootstrap et smoke tests. Il
decrit comment installer le projet et demarrer le socle Docker Compose sur une
station de travail ou sur le serveur IQA GPU RTX 3060.

## 1. Prerequis

- Ubuntu Server (cible officielle) ou poste de dev Linux/WSL2.
- `uv` installe (https://docs.astral.sh/uv/).
- Docker + Docker Compose plugin v2.
- Sur le serveur IQA GPU : driver NVIDIA, NVIDIA Container Toolkit et extra
  `cu128` pour les commandes PyTorch GPU.
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
`tests/contracts/test_architecture_contract.py` et
`tests/contracts/test_repo_contract.py`).

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

- Prometheus scrape `iqa-api:8000/metrics`, `iqa-inference:8100/metrics`,
  Airflow (via `statsd-exporter:9102`), MinIO et lui-meme (config
  `deploy/prometheus/prometheus.yml`). Cibles : `http://localhost:9090/targets`.
- Grafana est accessible derriere `reverse-proxy` (`/grafana/`), provisioning
  dans `deploy/grafana/provisioning/`. Le dashboard `IQA - Vue d'ensemble`
  (dossier IQA) est charge automatiquement : distribution Vert/Orange/Rouge,
  latence predict, erreurs, ROI fail, incidents IA, modele actif et
  disponibilite/verrou GPU.

## 7. Airflow (LocalExecutor, pool GPU)

```bash
cd deploy
docker compose --env-file ../.env up airflow-init   # une fois : db migrate + import pools.json
docker compose --env-file ../.env run --rm airflow-webserver airflow users create \
  --username admin \
  --firstname IQA \
  --lastname Admin \
  --role Admin \
  --email admin@example.local \
  --password <mot-de-passe-admin-airflow>
docker compose --env-file ../.env up -d airflow-webserver airflow-scheduler
```

- `AIRFLOW__CORE__EXECUTOR=LocalExecutor` (pas de Celery/Kubernetes).
- Le pool `iqa_gpu` (1 slot) est importe depuis `deploy/airflow/pools.json` et
  contraint les taches GPU du DAG `iqa_lifecycle` a `max_active_tasks=1`.
- Les DAGs Phase 1 sont importables et exposent les bonnes frontieres batch ;
  certaines commandes appelees restent des squelettes avec statut `planned`.
- Ne pas commiter le mot de passe Airflow ; il reste un secret d'exploitation.

### Verrou GPU (pas de train pendant la demo)

Le serveur n'a qu'un GPU (RTX 3060). Au-dela du pool Airflow, un verrou fichier
partage (`IQA_GPU_LOCK_PATH`, volume `gpu_lock`) serialise inference et
entrainement sur l'hote :

- Avant la demo, mettre `IQA_GPU_DEMO_HOLD=1` dans `.env` : `iqa-inference`
  prend le verrou au demarrage et le garde pendant toute la demo.
- Un `iqa-trainer` lance pendant ce temps est refuse immediatement (sortie 75 ;
  utiliser `--wait-for-gpu` pour attendre la liberation, `--no-gpu-lock` pour un
  dry-run CPU). Etat visible via `iqa_inference_gpu_lock_held` (dashboard).

Sur serveur RTX 3060, les tests d'inference/training GPU utilisent `cu128` :

```bash
docker compose --env-file ../.env -f docker-compose.yml -f docker-compose.gpu.yml up -d iqa-inference
docker compose --env-file ../.env -f docker-compose.yml -f docker-compose.gpu.yml exec iqa-inference \
  uv run --extra cu128 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 8. Streamlit (vitrine Sophie Phase 1)

```bash
docker compose up -d iqa-streamlit
```

Ouvre `http://localhost:8501` : modele actif (`/model/version`), lots
(`/replay-scenarios`), statut piece (`/piece-events/{id}/predict`) et
formulaire feedback `oracle_gt` (`/feedback`). C'est une vitrine MVP ; aucun
historique PostgreSQL n'est encore branche en Phase 1.

Le formulaire de feedback montre le parcours cible. Le workflow operationnel du
MVP reste automatise par `oracle_gt`; `human_sophie` est futur.

## 9. CI

`.github/workflows/ci.yml` execute `ruff check` puis `pytest -q` sur chaque
push/PR vers `main`. La CI ne declenche jamais d'entrainement (decision
Phase 1, cf. `architecture-iqa.md` section 13).

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
