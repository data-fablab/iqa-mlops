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

## 5.2 Preuve Airflow container runtime

La preuve Phase 3 Airflow verifie que les DAGs metier orchestrent des conteneurs
via `DockerOperator`, sans importer le runtime metier dans le scheduler. Docker
Compose orchestre les services longs ; Airflow orchestre les workflows metier
applicatifs. Le DAG `iqa_lifecycle` lance le pipeline applicatif Feature-AE de
reference avec `iqa-run-replay-lifecycle-cycle`.

```bash
uv run --extra cpu iqa-check-airflow-container-runtime --json

docker compose exec airflow-webserver airflow dags list
docker compose exec airflow-webserver airflow dags list-import-errors
docker compose exec airflow-webserver airflow pools list
docker compose exec airflow-webserver airflow dags unpause iqa_dvc_reproducibility
docker compose exec airflow-webserver airflow dags unpause iqa_lifecycle_trigger
docker compose exec airflow-webserver airflow dags unpause iqa_drift_piece_a_p4
docker compose exec airflow-webserver airflow dags unpause iqa_lifecycle
docker compose exec airflow-webserver airflow dags trigger iqa_dvc_reproducibility \
  --conf '{"with_network": false,"skip_regeneration": true}'
docker compose exec airflow-webserver airflow dags trigger iqa_lifecycle_trigger \
  --conf '{"scenario_id":"production_replay_natural","conforming_validated_count":50,"drift_confirmed":false,"roi_fail_rate":0.0}'
docker compose exec airflow-webserver airflow dags trigger iqa_drift_piece_a_p4
docker compose exec airflow-webserver airflow dags trigger iqa_lifecycle \
  --conf '{"mode":"progressive-train","max_events":260,"lifecycle_interval":50,"max_cycles":3,"epochs":10,"target_stage":"test","promotion_min_delta":0.0}'
```

Le backend Docker est valide pour la Phase 3. Le socket Docker
`/var/run/docker.sock` reste un privilege fort a reserver au serveur MVP de
confiance ; Kubernetes reste Phase 4. Le lifecycle reste declenche par evenement
data ou manuellement par un operateur Airflow, et il n'y a pas de training via
CI. MLflow Registry est la source de verite modele pour le stage `test`; MinIO
stocke les checkpoints et artefacts.

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

Kong est la cible Phase 3 pour la couche API Gateway / policies : protection de
routes, extension d'authentification, rate limiting et gouvernance transversale.
Nginx reste le reverse proxy operationnel/fallback du compose tant que Kong est
integre progressivement. `iqa-api` reste l'API metier ; Kong porte les politiques
transverses ; Nginx assure l'exposition pragmatique des services.

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

La CI (`publish-images`) builde et pousse les images par role vers Docker Hub
avec des tags **immuables** (SHA git pour chaque push ; tag de version `vX.Y.Z`
sur un tag git `v*`). Jamais de `latest`. Le tag recommande pour la preuve
serveur Phase 3 est le tag SHA CI : `IQA_IMAGE_TAG=sha-<commit>`.

Le job est opt-in. Avant de compter sur Docker Hub, verifier/configurer les
prerequis GitHub Actions :

```bash
gh variable list
gh secret list
gh variable set IQA_PUBLISH_IMAGES --body true
gh variable set IQA_IMAGE_REGISTRY --body <namespace-dockerhub>
gh secret set DOCKERHUB_USERNAME --body <user>
gh secret set DOCKERHUB_TOKEN
```

`DOCKERHUB_TOKEN` doit etre un access token Docker Hub avec droit `write`.
Les images publiees couvrent `iqa-serving`, `iqa-ml`, `iqa-data`,
`iqa-dvc-gate` et l'image custom `iqa-airflow`.

Sur le serveur, l'overlay `docker-compose.prod.yml` tire les images du registre au
lieu de builder localement. Figer la version a deployer dans `.env` :

```bash
# .env
IQA_IMAGE_REGISTRY=<namespace-dockerhub>
IQA_IMAGE_TAG=sha-<commit>     # tag immuable publie par la CI ; jamais latest
IQA_DOCKER_GID=<gid-du-socket-docker>
```

Le serveur s'authentifie sur Docker Hub avec `docker login`, puis tire les
images en HTTPS. SSH ne sert pas a connecter Docker Hub au serveur ; SSH ne
serait utile que pour un futur deploiement distant automatise depuis GitHub
Actions.

```bash
docker login
docker pull <namespace-dockerhub>/iqa-airflow:sha-<commit>
docker pull <namespace-dockerhub>/iqa-serving:sha-<commit>
docker pull <namespace-dockerhub>/iqa-ml:sha-<commit>
docker pull <namespace-dockerhub>/iqa-data:sha-<commit>
docker pull <namespace-dockerhub>/iqa-dvc-gate:sha-<commit>

cd deploy
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
bash ../deploy/smoke-test.sh
```

Rollback applicatif = redeployer le tag precedent (`IQA_IMAGE_TAG=v0.0.9`, `pull`,
`up -d`). Procedure detaillee (cas partiel, base de donnees, validation) :
`rollback-server.md`. Le rollback **modele** reste distinct : il passe par le
MLflow Registry (`rollback.md`).

### 8.2 Demo reproductible from scratch

```bash
bash deploy/demo-from-scratch.sh      # ou : make demo-scratch
```

Repart d'un etat vierge (`down -v`), redemarre toute la stack, attend l'API,
joue le smoke test puis le parcours metier (`iqa-demo-phase2`). Ideal pour une
soutenance ou une validation de bout en bout. `--yes` saute la confirmation du
`down -v` destructif.

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
