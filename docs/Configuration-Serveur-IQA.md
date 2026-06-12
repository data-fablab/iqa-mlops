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
| Latence cible p95 < 1 seconde | GPU local + FastAPI | OK pour MVP |
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
| Reverse proxy | Nginx ou Traefik |

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
| FastAPI | API d'inference et contrats `/predict`, `/feedback`, `/health`, `/metrics`, `/admin/reload-model` |
| Streamlit | Interface Sophie/Marc et vitrine du workflow qualite |
| PyTorch CUDA | ROI segmenter fige, teacher ResNet18 fige, Feature-AE |
| MLflow | Tracking experiments, registry modele, comparaison candidats |
| PostgreSQL | Backend MLflow et metadata store applicatif obligatoire |
| MinIO | Stockage objet local S3-compatible |
| DVC | Versioning donnees avec remote `s3://iqa-dvc` |
| Prometheus | Collecte metriques API, modele, drift, systeme |
| Grafana | Dashboards supervision |
| Airflow | Orchestration ingestion, replay, lifecycle Feature-AE, monitoring |
| Nginx/Traefik | Routage HTTP et exposition propre des services |

Airflow est plus lourd que Prefect, mais plus lisible pour ce projet car les DAGs rendent visibles les dependances MLOps : ingestion, replay, dataset candidat, entrainement, evaluation, promotion et reload.

## 6. Services Docker Compose cibles

```text
iqa-api
iqa-streamlit
iqa-inference
iqa-trainer
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
iqa-heatmaps
iqa-models
iqa-backups
```

Le job `minio-init` cree ces buckets au demarrage via `mc`. Les credentials sont fournis par `.env`, jamais commit dans Git.

## 9. Strategie stockage et retention

Le SSD de 500 Go est suffisant pour le MVP, mais impose une discipline de retention.

| Artefact | Regle recommandee |
|---|---|
| Dataset source Casting | bucket `iqa-source-datasets`, immutable |
| Images brutes production/replay | bucket `iqa-ingested-images`, tracees par `sha256` et URI PostgreSQL |
| Manifests CSV | Conserver dans Git si taille raisonnable |
| DVC remote `iqa-dvc` | garder versions utiles, nettoyage DVC periodique |
| MLflow artifacts | bucket `mlflow-artifacts`, garder runs promus et candidats recents |
| Modeles | bucket `iqa-models`, garder `prod`, `previous_prod`, candidats archives |
| Heatmaps | bucket `iqa-heatmaps`, expiration automatique des lots, retention curated |
| Logs applicatifs | Rotation automatique |
| Prometheus | Retention courte a moyenne pour MVP |

Regle simple MVP :
```text
prod + previous_prod + 3 derniers candidats + rapports de validation
```

## 10. Repartition des ressources

| Service | CPU/RAM | GPU |
|---|---|---|
| FastAPI | Faible a moyen | Non ou indirect |
| Inference PyTorch | Moyen | Oui |
| Training Feature-AE | Moyen a eleve | Oui |
| Streamlit | Faible | Non |
| MLflow | Faible | Non |
| MinIO | Faible a moyen | Non |
| PostgreSQL | Faible a moyen | Non |
| Prometheus/Grafana | Faible | Non |
| Airflow webserver/scheduler | Moyen | Non |

Bonne pratique : eviter de lancer un entrainement Feature-AE lourd pendant une demonstration d'inference temps reel. Les DAGs Airflow doivent declencher les traitements longs de maniere controlee, avec `LocalExecutor` et une concurrence limitee.

## 11. Securite minimale

Pour un serveur local de demonstration :
- acces SSH avec cle ;
- mots de passe forts pour Grafana, MLflow si expose, PostgreSQL ;
- credentials MinIO dans `.env`, jamais dans Git ;
- buckets non publics par defaut ;
- URLs presignees courtes pour l'affichage des heatmaps si necessaire ;
- reverse proxy unique en entree ;
- ports internes non exposes si possible ;
- sauvegarde reguliere de PostgreSQL, MLflow, manifests et modeles promus ;
- separation claire entre donnees brutes, simulations et artefacts de production ;
- pas d'exposition Internet directe sans VPN ou tunnel controle.

## 12. Verdict

La station HP Z420 avec Xeon E5-1680 v2, 40 Go RAM et RTX 3060 12 Go est adaptee pour heberger le MVP MLOps IQA.

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
