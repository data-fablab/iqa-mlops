# Configuration serveur IQA

## 1. Objectif
Ce document decrit la configuration materielle et logicielle recommandee pour heberger le MVP MLOps **Industrial Quality Assistant** sur une station locale.

Le serveur doit couvrir :
- l'API d'inference ;
- l'interface Sophie/Marc ;
- le replay des lots ;
- l'inference GPU ROI + Feature-AE ;
- le reentrainement controle du Feature-AE ;
- le suivi MLflow ;
- le stockage objet MinIO ;
- le monitoring Prometheus/Grafana ;
- l'orchestration Airflow des pipelines ;
- le versioning et la tracabilite des donnees/modeles.

## 2. Configuration materielle cible

| Composant | Configuration |
|---|---|
| Machine | HP Z420 Workstation |
| Processeur | Intel Xeon E5-1680 v2 |
| CPU | 8 coeurs / 16 threads |
| Cache CPU | 25 Mo |
| Frequence | 3,00 GHz |
| Memoire | 40 Go RAM |
| GPU | NVIDIA RTX 3060 |
| VRAM | 12 Go |
| Stockage | SSD 500 Go |
| Usage cible | Serveur local MLOps complet |

## 3. Adequation avec le cadrage projet

| Besoin du projet | Configuration serveur | Statut |
|---|---|---|
| Inference visuelle locale | RTX 3060 12 Go | OK |
| Latence cible p95 < 1 seconde | GPU local + service inference | OK pour MVP |
| Feature-AE good-only | GPU + 40 Go RAM | OK |
| Teacher ResNet18 fige | GPU adapte | OK |
| Segmenteur ROI fige | Inference GPU uniquement | OK |
| Replay naturel et drift controle | CPU/RAM suffisants | OK |
| MLflow tracking + registry | SSD + volume persistant | OK avec retention |
| MinIO local | stockage objet S3-compatible | OK avec retention |
| Prometheus/Grafana | Services legers | OK |
| Streamlit Sophie | Service leger | OK |
| API FastAPI | Service leger | OK |
| Kubernetes | Hors perimetre MVP | Non requis |

La configuration est adaptee au MVP si l'architecture reste sur une station unique avec Docker Compose. Le point de vigilance principal est le stockage de 500 Go.

## 4. Systeme recommande

| Couche | Choix recommande |
|---|---|
| Systeme | Ubuntu Server 24.04 LTS |
| Conteneurisation | Docker Engine |
| Orchestration locale | Docker Compose |
| GPU containers | NVIDIA Container Toolkit |
| Driver GPU | Driver NVIDIA recent compatible RTX 3060 |
| Python | Python 3.11 ou 3.12 |
| Gestion dependances | uv |
| Acces distant | SSH |
| Reverse proxy | Nginx |

Ubuntu Server est la cible officielle de deploiement. Windows + WSL2 n'est pas retenu pour le serveur IQA, afin de garder une base plus stable, plus proche d'une exploitation serveur et plus simple a administrer a distance. Kubernetes n'est pas recommande pour ce MVP : il complexifie le projet sans benefice suffisant sur une station unique.

Airflow est retenu comme orchestrateur projet pour rendre la boucle MLOps explicite et demonstrable. Sur cette station, il doit rester en configuration legere :
```text
LocalExecutor
PostgreSQL metadata DB
pas de CeleryExecutor
pas de KubernetesExecutor
workers limites
DAGs courts et idempotents
```

## 5. Briques logicielles

| Brique | Role |
|---|---|
| FastAPI | API applicative et contrats `/predict`, `/feedback`, `/health`, `/metrics`, `/admin/reload-model` |
| Streamlit | Interface Sophie/Marc et vitrine du workflow qualite |
| PyTorch CUDA | ROI segmenter fige, teacher ResNet18 fige, Feature-AE |
| MLflow | Tracking experiments, registry modele, comparaison candidats |
| PostgreSQL | Une instance avec bases separees `iqa_metadata`, `mlflow`, `airflow` |
| MinIO | Stockage objet local S3-compatible |
| DVC | Versioning donnees avec remote `s3://iqa-dvc` |
| Prometheus | Collecte metriques API, modele, drift, systeme |
| Grafana | Dashboards supervision |
| Airflow | Orchestration ingestion, replay, lifecycle Feature-AE, monitoring |
| Nginx | Routage HTTP et exposition propre des services |

Airflow est plus lourd que Prefect, mais plus lisible pour ce projet car les
DAGs rendent visibles les dependances MLOps : ingestion, replay, dataset
candidat, entrainement, evaluation, promotion et reload. En Phase 1, ces DAGs
sont importables et initialisables ; plusieurs taches restent des frontieres
batch/squelettes avant le branchement complet PostgreSQL, MLflow et promotion.

## 6. Services Docker Compose du repo

```text
iqa-api
iqa-inference
iqa-streamlit
iqa-ingestion
iqa-replay
iqa-trainer
iqa-monitoring
airflow-init
airflow-webserver
airflow-scheduler
mlflow
minio
minio-init
postgres
prometheus
grafana
reverse-proxy
```

`docker-compose.gpu.yml` active l'extra serveur `cu128` pour `iqa-inference` et
`iqa-trainer`. `docker-compose.timezone.yml` ajoute les overrides de fuseau
horaire Europe/Paris pour l'exploitation serveur.

Services optionnels selon avancement :
```text
node-exporter
cadvisor
backup
```

## 7. Ports recommandes

| Service | Port interne |
|---|---:|
| FastAPI | 8000 |
| Inference PyTorch | 8100 |
| Streamlit | 8501 |
| MLflow | 5000 |
| MinIO API | 9000 |
| MinIO Console | 9001 |
| Grafana | 3000 |
| Prometheus | 9090 |
| PostgreSQL | 5432 |
| Airflow | 8080 |

Exposition via reverse proxy :

```text
/iqa      -> Streamlit
/api      -> FastAPI
/mlflow   -> MLflow
/minio    -> MinIO console, acces admin restreint
/grafana  -> Grafana
/airflow  -> Airflow
```

## 8. Organisation des volumes

Arborescence recommandee :

```text
/opt/iqa
  /data
  /datasets
  /models
  /artifacts
  /mlruns
  /minio
  /logs
  /postgres
  /airflow
  /prometheus
  /grafana
  /backups
```

Correspondance projet :

| Volume | Contenu |
|---|---|
| `/opt/iqa/data` | donnees sources et manifests |
| `/opt/iqa/datasets` | datasets candidats Feature-AE |
| `/opt/iqa/models` | modeles promus et archives |
| `/opt/iqa/artifacts` | heatmaps, exports, evaluations |
| `/opt/iqa/mlruns` | artefacts MLflow |
| `/opt/iqa/minio` | buckets S3 locaux : DVC, MLflow, heatmaps, modeles, backups |
| `/opt/iqa/logs` | logs API, inference, training |
| `/opt/iqa/postgres` | base PostgreSQL |
| `/opt/iqa/airflow` | DAGs, logs Airflow, metadata de runtime |
| `/opt/iqa/prometheus` | time series monitoring |
| `/opt/iqa/grafana` | dashboards et configuration |
| `/opt/iqa/backups` | sauvegardes minimales |

Buckets MinIO :
```text
iqa-source-datasets
iqa-dvc
iqa-ingested-images
mlflow-artifacts
iqa-roi-masks
iqa-heatmaps
iqa-gt-masks
iqa-models
iqa-backups
```

Le job `minio-init` cree ces buckets au demarrage via `mc`. Les credentials sont fournis par `.env`, jamais commit dans Git.

## 9. Strategie stockage et retention

Le SSD de 500 Go est suffisant pour le MVP, mais impose une discipline de retention.

| Artefact | Regle recommandee |
|---|---|
| Dataset source Casting | bucket `iqa-source-datasets`, immutable |
| Images brutes production/replay | bucket `iqa-ingested-images`, tracees par `sha256` ; URI PostgreSQL cible |
| Manifests CSV | Conserver dans Git si taille raisonnable |
| DVC remote `iqa-dvc` | garder versions utiles, nettoyage DVC periodique |
| MLflow artifacts | bucket `mlflow-artifacts`, garder runs promus et candidats recents |
| Modeles | bucket `iqa-models`, artefacts candidats, promus et archives ; statut prod dans MLflow |
| Masques ROI | bucket `iqa-roi-masks`, masques produits par le segmenteur ROI fige |
| Heatmaps | bucket `iqa-heatmaps`, expiration automatique des lots, retention curated |
| Masques GT/oracle | bucket `iqa-gt-masks`, preuve feedback et calibration |
| Logs applicatifs | Rotation automatique |
| Prometheus | Retention courte a moyenne pour MVP |

Regle simple MVP :
```text
prod MLflow + version rollback MLflow + 3 derniers candidats + rapports de validation
```

## 10. Repartition des ressources

| Service | CPU/RAM | GPU |
|---|---|---|
| FastAPI API | Faible a moyen | Non |
| Inference PyTorch | Moyen | Oui |
| Training Feature-AE | Moyen a eleve | Oui |
| Ingestion/replay/monitoring batch | Faible a moyen | Non sauf calculs optionnels |
| Streamlit | Faible | Non |
| MLflow | Faible | Non |
| MinIO | Faible a moyen | Non |
| PostgreSQL | Faible a moyen | Non |
| Prometheus/Grafana | Faible | Non |
| Airflow webserver/scheduler | Moyen | Non |

Bonne pratique : l'inference et le training sont des services separes, mais ils
partagent la meme RTX 3060. Airflow doit utiliser un pool ou verrou GPU avec
`max_active_tasks=1`. Pendant une demonstration, `iqa-inference` est prioritaire
et les entrainements lourds doivent etre suspendus ou controles.

## 11. Securite minimale

Pour un serveur local de demonstration :
- acces SSH avec cle ;
- mots de passe forts pour Grafana, MLflow si expose, PostgreSQL ;
- credentials MinIO dans `.env`, jamais dans Git ;
- buckets non publics par defaut ;
- URLs presignees courtes pour l'affichage des heatmaps si necessaire ;
- reverse proxy unique en entree ;
- ports internes non exposes si possible ;
- sauvegarde reguliere de la base `iqa_metadata`, des manifests et des modeles promus ;
- separation claire entre donnees brutes, simulations et artefacts de production ;
- pas d'exposition Internet directe sans VPN ou tunnel controle.

## 12. Verdict

Le serveur IQA GPU RTX 3060, base sur la station HP Z420 avec Xeon E5-1680 v2 et 40 Go RAM, est adapte pour heberger le MVP MLOps IQA.

Elle permet de faire tourner :
- le replay des scenarios ;
- l'API d'inference ;
- l'interface Sophie ;
- l'inference GPU ;
- le reentrainement controle du Feature-AE ;
- MLflow ;
- MinIO ;
- Prometheus/Grafana ;
- les DAGs Airflow d'orchestration locale.

La recommandation finale est :

```text
Ubuntu Server 24.04 LTS
+ Docker Compose
+ NVIDIA Container Toolkit
+ FastAPI / Streamlit
+ PyTorch CUDA
+ MLflow / PostgreSQL
+ MinIO / DVC remote S3
+ Prometheus / Grafana
+ Airflow LocalExecutor
```

Cette configuration est coherente avec le cadrage projet, a condition de conserver Kubernetes hors perimetre et de mettre en place une politique de retention des artefacts.

Decision explicite : la procedure officielle de deploiement est `Ubuntu Server -> Docker Compose -> dvc pull -> docker compose up`. Windows + WSL2 peut rester un environnement de developpement individuel, mais pas la cible serveur retenue.

## 13. Procedure serveur MVP

Initialisation du repo :

```bash
cd /opt/iqa
git clone https://github.com/data-fablab/iqa-mlops.git
cd iqa-mlops
uv sync --extra cpu --extra data
uv run --extra cpu --extra data ruff check src scripts tests
uv run --extra cpu --extra data pytest -q
```

Sur le serveur RTX 3060, les tests et taches data restent lances avec
`--extra cpu --extra data`. Les services ou commandes qui executent PyTorch sur
GPU doivent utiliser `--extra cu128`.

Configuration environnement :

```bash
cp .env.example .env
nano .env
```

Le fichier `.env` serveur doit remplacer tous les secrets `change-me-*` :

```text
IQA_POSTGRES_PASSWORD
MINIO_ROOT_USER
MINIO_ROOT_PASSWORD
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
GF_SECURITY_ADMIN_PASSWORD
IQA_ADMIN_TOKEN
IQA_SERVICE_TOKEN
```

MLflow valide le header `Host` pour limiter les attaques DNS rebinding. Les
adresses exposees doivent donc etre listees avec et sans port dans :

```text
IQA_MLFLOW_ALLOWED_HOSTS
```

Exemple pour le serveur :

```text
localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,<SERVER_TAILSCALE_IP>,<SERVER_TAILSCALE_IP>:5000,mlflow,mlflow:5000
```

Demarrage socle :

```bash
cd /opt/iqa/iqa-mlops/deploy
docker compose --env-file ../.env up -d postgres minio minio-init mlflow prometheus grafana
docker compose ps
```

Initialisation Airflow :

```bash
cd /opt/iqa/iqa-mlops/deploy
docker compose --env-file ../.env up airflow-init
docker compose --env-file ../.env run --rm airflow-webserver airflow users create \
  --username admin \
  --firstname IQA \
  --lastname Admin \
  --role Admin \
  --email admin@example.local \
  --password <mot-de-passe-admin-airflow>
docker compose --env-file ../.env up -d airflow-webserver airflow-scheduler
```

Les services Airflow utilisent la base PostgreSQL `airflow` via
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`. Le service `airflow-init` est la
procedure officielle d'initialisation : il execute `airflow db migrate` et
importe `deploy/airflow/pools.json`, dont le pool GPU `iqa_gpu`. Ne pas lancer
une initialisation Airflow hors compose, sinon la base par defaut du conteneur
peut etre utilisee a la place.

Etat serveur actuel : Airflow est installe et demarrable via
`airflow-webserver` et `airflow-scheduler`. Les DAGs Phase 1 sont importables ;
plusieurs taches restent des frontieres batch/squelettes avant le branchement
complet PostgreSQL, MLflow et promotion modele.

Initialisation DVC cote serveur :

```bash
cd /opt/iqa/iqa-mlops
uv run --extra cpu --extra data dvc remote modify --local iqa-minio endpointurl http://localhost:9000
uv run --extra cpu --extra data dvc remote modify --local iqa-minio access_key_id "$MINIO_ROOT_USER"
uv run --extra cpu --extra data dvc remote modify --local iqa-minio secret_access_key "$MINIO_ROOT_PASSWORD"
uv run --extra cpu --extra data dvc pull
```

Demarrage API et inference CPU/smoke :

```bash
cd /opt/iqa/iqa-mlops/deploy
docker compose --env-file ../.env up -d iqa-api iqa-inference reverse-proxy
curl http://localhost/api/health
curl http://localhost:8100/health
```

Demarrage du stack avec inference/trainer GPU RTX 3060 (`cu128`) :

```bash
cd /opt/iqa/iqa-mlops/deploy
docker compose --env-file ../.env -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Validation PyTorch CUDA dans le conteneur inference :

```bash
docker compose --env-file ../.env -f docker-compose.yml -f docker-compose.gpu.yml exec iqa-inference uv run --extra cu128 python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Generation ROI bootstrap V0 sur serveur RTX 3060 :

Le checkpoint ROI officiel est reference par manifest Git et stocke dans MinIO :
`s3://iqa-models/roi_segmenter_v001_fixed/checkpoint.pt`. Les checkpoints lourds
ne doivent pas etre stockes dans Git ni sous `models/`. Restaurer les artefacts
dans le cache local controle avant execution :

```bash
uv run --extra cpu iqa-restore-model-artifacts --model-version roi_segmenter_v001_fixed
```

```bash
cd /opt/iqa/iqa-mlops
uv run --extra cu128 --extra data iqa-generate-bootstrap-roi \
  --manifest data/metadata/feature_ae_bootstrap_events.csv \
  --image-root data/raw/hss-iad \
  --output-dir data/processed/roi/bootstrap_v001 \
  --roi-model-version roi_segmenter_v001_fixed \
  --device cuda
```

La sortie locale `data/processed/roi/bootstrap_v001` reste hors Git. En cible
production/replay, ces masques sont stockes dans `s3://iqa-roi-masks` et seuls
les URI/faits sont conserves dans PostgreSQL.

Validation GPU Docker :

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

Note : le compose principal reste CPU pour etre portable. Le fichier
`docker-compose.gpu.yml` active CUDA uniquement pour les services qui en ont
besoin : `iqa-inference` et `iqa-trainer`. Sur le serveur IQA, l'extra CUDA de
reference est `cu128`.

## 14. Decisions de convergence infrastructure

Le pyproject.toml racine reste conserve pour le repo initial. L'isolation fine
se fait d'abord par Docker Compose ; une migration `services/` est reportee.

`iqa-api` et `iqa-inference` restent deux services separes dans l'architecture
cible. Airflow orchestre par contrats HTTP ou commandes batch.

PostgreSQL est une seule instance avec trois bases logiques :

```text
iqa_metadata
mlflow
airflow
```

MLflow Registry est la source de verite cible de la promotion et du rollback.
MinIO stocke les artefacts, sans prefixe S3 `prod` faisant autorite.

Les faits metier portent `event_time`, `recorded_at` et `is_simulated` afin de
separer le temps rejoue, le temps systeme et la nature simulation/production.
