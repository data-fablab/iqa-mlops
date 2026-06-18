# Deploy Runbook - IQA (serveur)

Runbook de deploiement du stack IQA sur le serveur cible (Ubuntu Server, GPU
RTX 3060). Complete `runbook-phase1-iqa.md` (install locale / dev) en couvrant
la mise en service complete, les smoke tests, l'observabilite et le rollback.

Pour la cartographie buckets / DVC / PostgreSQL et les politiques de retention,
voir `retention_storage.md`.

## 1. Cible et prerequis

- Ubuntu Server (cible officielle ; Windows/WSL2 non retenu).
- Docker + plugin Docker Compose v2.
- Pour le GPU : driver NVIDIA + `nvidia-container-toolkit` (runtime `cu128`).
- `uv` uniquement si on lance des commandes hors conteneurs.
- Acces reseau aux ports publies (ou via le reverse proxy uniquement).

Verifier le GPU cote hote :

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 2. Recuperation du code

```bash
git clone https://github.com/data-fablab/iqa-mlops.git
cd iqa-mlops
```

## 3. Secrets et configuration

```bash
cp .env.example .env
```

Renseigner au minimum, dans `.env` (jamais commite) :

- `IQA_POSTGRES_USER` / `IQA_POSTGRES_PASSWORD`
- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`
- `IQA_S3_ACCESS_KEY_ID` / `IQA_S3_SECRET_ACCESS_KEY` (= identifiants MinIO)
- `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD`
- `IQA_ADMIN_TOKEN` / `IQA_SERVICE_TOKEN`
- `IQA_MLFLOW_ALLOWED_HOSTS` (ajouter l'IP/host du serveur)
- `IQA_GPU_DEMO_HOLD=1` pendant une demo (verrou GPU), `0` sinon

## 4. Demarrage du stack

Toutes les commandes `docker compose` se lancent depuis `deploy/`.

```bash
cd deploy
```

### 4.1 Socle (donnees + objet)

```bash
docker compose up -d postgres minio minio-init
docker compose logs minio-init   # verifie creation buckets + ILM heatmaps
```

PostgreSQL provisionne 3 bases (`iqa_metadata`, `mlflow`, `airflow`).

### 4.2 Application (API + inference)

CPU :

```bash
docker compose up -d iqa-inference iqa-api
```

GPU (RTX 3060, runtime `cu128`) via l'overlay :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d iqa-inference iqa-api
```

### 4.3 MLflow

```bash
docker compose up -d mlflow
```

### 4.4 Observabilite + reverse proxy

```bash
docker compose up -d statsd-exporter prometheus grafana reverse-proxy
```

### 4.5 Airflow (LocalExecutor + pool GPU)

```bash
docker compose up airflow-init   # une fois : db migrate + import pools.json
docker compose up -d airflow-webserver airflow-scheduler
```

Le pool `iqa_gpu` (1 slot) contraint les taches GPU a `max_active_tasks=1`.

> **Privilege fort (a assumer / cantonner).** En backend `docker` (ADR 0008), le
> `airflow-scheduler` monte le socket Docker hote (`/var/run/docker.sock`) pour
> lancer les conteneurs de tache. Qui accede au socket est root-equivalent sur
> l'hote : a reserver au MVP local/serveur de confiance, jamais expose publiquement.
> Durcissement possible sans changer les DAGs : `docker-socket-proxy` (filtre l'API
> Docker) via `IQA_DOCKER_URL=tcp://...`. La cible Kubernetes (`IQA_AIRFLOW_BACKEND=k8s`)
> supprime ce privilege (l'orchestration passe par l'API server + RBAC).

### 4.6 Vitrine Streamlit (Accueil + Marc + Sophie)

```bash
docker compose up -d iqa-streamlit
```

## 5. Smoke tests post-deploiement

Depuis la racine du repo :

```bash
bash deploy/smoke-test.sh
```

Verifie de bout en bout :

- **Applicatif** : API + inference (`/health`, `/metrics`, `/model/version`,
  `/predictions`, `/lots/summary`).
- **Donnees / artefacts** : MinIO (live), MLflow.
- **Observabilite** : Prometheus (+ targets), Grafana.
- **Orchestration** : Airflow (`/health`). `iqa-monitoring` est un job batch
  (sans HTTP) ; sa couverture passe par Airflow + le job Prometheus `airflow`.
- **Gateway** : routage via le reverse-proxy (port 80) vers api, grafana,
  airflow, mlflow.

Code de sortie 0 = tout vert. Cibles surchargeables par variables
d'environnement (`IQA_AIRFLOW_URL`, `IQA_GATEWAY_URL`, etc. ; voir l'en-tete du
script).

## 5.1 Gate DVC / data lineage

Airflow expose le DAG `iqa_dvc_reproducibility` pour valider le remote DVC et la
reproductibilite metadata. Le mode reseau MinIO doit rester explicite :

```bash
docker compose exec airflow-webserver airflow dags trigger iqa_dvc_reproducibility \
  --conf '{"with_network": true}'
```

Ce DAG est un gate de reproductibilite ; il ne remplace pas les DAGs metier et ne
declenche pas de lifecycle modele.

## 6. Acces via le reverse proxy

Le service `reverse-proxy` (Nginx) expose tout derriere le port 80 :

```text
/api/        -> iqa-api:8000
/iqa/        -> iqa-streamlit:8501
/mlflow/     -> mlflow:5000
/minio/      -> minio:9001 (console)
/grafana/    -> grafana:3000
/airflow/    -> airflow-webserver:8080
```

Dashboard Grafana : dossier "IQA" -> `IQA - Vue d'ensemble` (V/O/R, latence,
erreurs, ROI fail, incidents IA, modele actif, verrou GPU).

## 7. Verrou GPU pendant une demo

Mono-GPU : pas d'entrainement concurrent pendant l'inference demo.

- `IQA_GPU_DEMO_HOLD=1` -> `iqa-inference` prend le verrou au demarrage et le
  garde toute la demo (volume partage `gpu_lock`).
- Un `iqa-trainer` lance pendant ce temps est refuse (sortie 75). Etat visible
  via la metrique `iqa_inference_gpu_lock_held` (dashboard).

## 8. Mise a jour

```bash
git pull
cd deploy
docker compose build iqa-api iqa-inference        # rebuild image applicative
docker compose up -d iqa-api iqa-inference
bash ../deploy/smoke-test.sh
```

Le lifecycle modele n'est jamais declenche par la CI ni par un deploiement : il
reste un evenement donnees (DAG Airflow). Le rollback modele se fait via le
MLflow Registry (source de verite), pas par redeploiement de conteneur.

### 8.1 Deploiement depuis les images publiees (registre, tags figes)

La CI (`publish-images`) builde et pousse les 3 images par role vers Docker Hub
avec des tags **immuables** (SHA git pour chaque push ; tag de version `vX.Y.Z`
sur un tag git `v*`). Jamais de `latest`. Activation : variable repo
`IQA_PUBLISH_IMAGES=true` + secrets `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`.

Sur le serveur, l'overlay `docker-compose.prod.yml` tire les images du registre au
lieu de builder localement. Figer la version a deployer dans `.env` :

```bash
# .env
IQA_IMAGE_REGISTRY=data-fablab
IQA_IMAGE_TAG=v0.1.0           # tag immuable publie par la CI ; jamais latest

cd deploy
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
bash ../deploy/smoke-test.sh
```

Rollback applicatif = redeployer le tag precedent (`IQA_IMAGE_TAG=v0.0.9`, `pull`,
`up -d`). Le rollback **modele** reste distinct : il passe par le MLflow Registry.

## 9. Sauvegardes

- PostgreSQL : `docker compose exec postgres pg_dumpall -U "$IQA_POSTGRES_USER"`
  vers `s3://iqa-backups`.
- MinIO : les buckets sont la source ; repliquer `s3://iqa-backups` hors site.
- Politiques de retention detaillees : voir `retention_storage.md`.

## 10. Arret

```bash
docker compose down          # stoppe, conserve les volumes nommes
docker compose down -v       # repart d'un etat vide (DANGER : efface donnees)
```

Les volumes `postgres_data`, `minio_data`, `gpu_lock` sont conserves entre les
demarrages sauf `down -v`.
