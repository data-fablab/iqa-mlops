# Architecture projet IQA

## 1. Principe

Cette architecture reprend l'esprit du template `ssime-git/mlops-project-template` :
- separation claire API / entrainement / monitoring / configuration / deploiement ;
- Docker Compose pour la station unique ;
- DVC avec remote S3 MinIO pour les donnees et datasets candidats ;
- MLflow pour les experiences et le registre de modeles ;
- MinIO comme stockage objet local pour DVC, MLflow artifacts, heatmaps, modeles et backups ;
- Prometheus/Grafana pour l'observabilite ;
- Airflow pour rendre la boucle MLOps visible et pilotable.

Elle integre les decisions de la proposition Ken retenues par l'equipe :
- `validation_set_v001` fige avant tout replay, hors calibration ;
- interface Sophie vitrine pour le MVP, avec feedback automatise par oracle GT ;
- PostgreSQL comme metadata store cible ;
- PostgreSQL stocke les faits, statuts et URI, jamais les fichiers lourds ;
- scenarios isoles par `scenario_id` ;
- stockage S3 local via MinIO, avec un client unique `src/iqa/storage` ;
- images entrantes stockees dans `s3://iqa-ingested-images` ;
- deux boucles separees : CI code et lifecycle modele Airflow ;
- deploiement officiel sur Ubuntu Server, pas Windows + WSL2.

## 2. Architecture logique

```text
Sophie / Marc
    |
    v
Streamlit
    |
    v
Reverse proxy
    |
    v
FastAPI `iqa-api`
    |
    +--> service `iqa-inference`
    |       -> ROI segmenter fige
    |       -> controle qualite ROI
    |       -> teacher ResNet18 fige
    |       -> Feature-AE actif
    |       -> score + heatmap + decision Vert/Orange/Rouge
    |
    +--> feedback
    |       -> oracle_gt apres prediction
    |       -> human_sophie cible future, non operationnel MVP
    |
    +--> PostgreSQL metadata
    |       -> predictions, feedback, lots, versions, incidents
    |
    +--> Prometheus metrics

Airflow
    |
    +--> service batch `iqa-ingestion`
    +--> service batch `iqa-replay`
    +--> service batch `iqa-trainer`
    +--> service batch `iqa-monitoring`
    |
    +--> DVC remote s3://iqa-dvc
    +--> MinIO artifact store
    +--> MLflow tracking + registry
    +--> /admin/reload-model
```

## 3. Simulation vs production reelle

Le dataset Casting est utilise comme historique industriel rejoue. Il sert a simuler un flux de production sans pretendre que les fichiers du repo sont le stockage usine final.

Flux cible production :
```text
camera / poste qualite / MES
-> production_ingest
-> MinIO s3://iqa-ingested-images
-> PostgreSQL piece_event + image_uri + contexte lot
-> FastAPI `iqa-api`
-> service `iqa-inference`
-> prediction + feedback
```

Flux MVP ecole :
```text
dataset Casting historique
-> historical_replay
-> meme contrat ingestion
-> MinIO/DVC URI compatible
-> PostgreSQL piece_event + image_uri + scenario_id
-> FastAPI `iqa-api`
-> service `iqa-inference`
-> prediction + feedback
```

Cette separation permet de remplacer la source de donnees plus tard sans changer les contrats runtime.

## 4. Arborescence cible

```text
iqa-mlops/
|-- README.md
|-- docs/
|   |-- cadrage.md
|   |-- architecture.md
|   |-- server-config.md
|   |-- PRD-IQA-MVP.md
|   |-- CONTEXT.md
|   |-- simulation_report.md
|   |-- validation_report.md
|   `-- adr/
|       |-- 0001-validation-set-fige-hors-replay.md
|       `-- 0002-airflow-comme-orchestrateur.md
|-- configs/
|   |-- paths.yaml
|   |-- replay_scenarios.yaml
|   |-- monitoring_thresholds.yaml
|   `-- promotion_gates.yaml
|-- airflow/
|   |-- dags/
|   |   |-- iqa_ingestion.py
|   |   |-- iqa_replay.py
|   |   |-- iqa_lifecycle.py
|   |   `-- iqa_monitoring.py
|   |-- plugins/
|   `-- requirements-airflow.txt
|-- data/
|   |-- raw/
|   |-- metadata/
|   |-- processed/
|   |-- model_datasets/
|   `-- validation/
|-- src/iqa/
|   |-- ingestion/
|   |-- replay/
|   |-- roi/
|   |-- models/
|   |-- inference/
|   |-- feedback/
|   |-- datasets/
|   |-- training/
|   |-- monitoring/
|   |-- storage/
|   |-- registry/
|   `-- api/
|-- models/
|-- reports/
|-- tests/
|   |-- dags/
|   |-- contracts/
|   |-- integration/
|   `-- ml/
|-- notebooks/
|-- deploy/
|   |-- docker-compose.yml
|   |-- grafana/
|   |-- minio/
|   |   |-- init-buckets.sh
|   |   `-- lifecycle-heatmaps.json
|   |-- prometheus/
|   `-- nginx/
|-- .env.example
|-- .github/workflows/
|-- Makefile
|-- pyproject.toml
|-- .gitignore
`-- .dvc/
    `-- config
```

## 5. Documents projet

Documents cibles :
- `cadrage.md` : vision, perimetre, donnees, scenarios, monitoring, gates ;
- `PRD-IQA-MVP.md` : exigences produit, user stories, decisions d'implementation ;
- `CONTEXT.md` : glossaire court et langage commun ;
- `architecture.md` : structure technique et responsabilites ;
- `Modele-Feature-AE-IQA.md` : contrat du modele vivant, training, evaluation, checkpoints ;
- `Modele-Segmentation-ROI-IQA.md` : contrat du segmenteur ROI fige et usage downstream ;
- `server-config.md` : configuration Z420 Ubuntu Server ;
- `adr/0001` : validation set fige hors replay ;
- `adr/0002` : Airflow comme orchestrateur ;
- `adr/0003` : MinIO comme stockage objet local.

Le PRD et les ADR evitent que les choix importants restent implicites dans le cadrage.

## 6. DAGs Airflow

Les DAGs orchestrent des modules du package `src/iqa`. Ils ne contiennent pas toute la logique metier.

### `iqa_ingestion.py`

```text
historical_replay ou production_ingest
-> inventory sha256
-> stockage image dans MinIO
-> piece_events = source_class + group_key
-> verification masques GT si historique
-> manifests metadata / PostgreSQL facts
```

Sorties :
- `casting_images_inventory.csv` ;
- `casting_piece_events.csv` ;
- controles de coherence.

### `iqa_replay.py`

```text
metadata + validation_set_v001
-> bootstrap hors replay
-> calibration_set_v001 hors replay
-> production_replay_natural
-> drift_domain_extension
-> lots horodates
```

Sorties :
- `feature_ae_bootstrap_events.csv` ;
- `casting_flux_replay_plan_natural.csv` ;
- `casting_flux_replay_plan_drift.csv` ;
- `replay_scenarios.csv`.

Invariant :
```text
bootstrap ∩ calibration ∩ replay ∩ validation = vide
```

### `iqa_lifecycle.py`

DAG vedette de la demonstration MLOps.

```text
monitoring/feedback
-> dataset candidat good-only ROI-ok
-> train Feature-AE candidat
-> evaluation validation_set_v001
-> quality gates
-> log MLflow
-> transition MLflow Registry vers prod si promotion
-> artefacts stockes dans s3://iqa-models
-> /admin/reload-model si promotion
```

Contraintes :
- aucun defaut dans le train normal ;
- aucune ROI warning/fail dans le train normal ;
- aucun piece event du validation set ou du calibration set dans le train ou le replay ;
- le ROI segmenter et le teacher restent figes.

### `iqa_monitoring.py`

```text
predictions par lot
-> drift teacher features
-> derive reconstruction Feature-AE
-> suivi ROI
-> alertes FN / drift / latence
-> trigger eventuel du DAG lifecycle
```

Le declenchement du lifecycle est un evenement donnees, pas un commit ni un cron systematique.

## 6. Donnees

```text
data/raw/            -> hss-iad versionne DVC
data/metadata/       -> manifests CSV
data/processed/      -> ROI, features, heatmaps, exports
data/model_datasets/ -> datasets candidats Feature-AE
data/validation/     -> validation_set_v001
data/metadata/       -> calibration_set_v001
```

Stockage objet MinIO :
```text
s3://iqa-source-datasets -> dataset historique immutable
s3://iqa-dvc             -> remote DVC
s3://iqa-ingested-images -> images brutes recues ou rejouees
s3://mlflow-artifacts  -> artefacts MLflow
s3://iqa-roi-masks     -> masques ROI produits par le segmenteur fige
s3://iqa-heatmaps      -> heatmaps et overlays
s3://iqa-models        -> artefacts modeles candidats, promus et archives
s3://iqa-backups       -> sauvegardes applicatives
```

Flux bootstrap et cycle normal :

```text
bootstrap_v001
-> feature_ae_bootstrap_events.csv
-> ROI segmenter fige
-> data/processed/roi/bootstrap_v001/roi_predictions.csv
-> dataset Feature-AE V0 good-only + ROI-ok

production_ingest / historical_replay
-> s3://iqa-ingested-images
-> ROI segmenter fige
-> s3://iqa-roi-masks
-> Feature-AE
-> s3://iqa-heatmaps
-> feedback oracle GT
-> faits et URI PostgreSQL
```

Manifests essentiels :
```text
casting_images_inventory.csv
casting_piece_events.csv
feature_ae_bootstrap_events.csv
casting_flux_replay_plan_natural.csv
casting_flux_replay_plan_drift.csv
replay_scenarios.csv
validation_set_v001.csv
calibration_set_v001.csv
```

## 7. Package `src/iqa`

```text
ingestion/   -> inventaire, sha256, piece_events
replay/      -> lots, cadence, scenarios
roi/         -> ROI segmenter fige, controle qualite ROI
models/      -> code PyTorch teacher + Feature-AE
inference/   -> pipeline prediction image/piece
feedback/    -> oracle GT MVP, vitrine Sophie, regles de priorite futures
datasets/    -> datasets candidats good-only
training/    -> train/eval/calibration/gates Feature-AE
monitoring/  -> drift, metriques, alertes
storage/     -> client S3 MinIO unique, URI logiques, URLs presignees
registry/    -> MLflow, promotion, rollback, model loading
api/         -> FastAPI routes et schemas
```

`src/iqa/models/` contient le code modele. Les artefacts entraines sont stockes dans MinIO, principalement `s3://iqa-models`.

Contrats detailles :
- [Modele Feature-AE IQA](Modele-Feature-AE-IQA.md) ;
- [Modele Segmentation ROI IQA](Modele-Segmentation-ROI-IQA.md).

## 8. Services Docker Compose

Services cibles :

```text
iqa-api
iqa-inference
iqa-streamlit
iqa-ingestion
iqa-replay
iqa-trainer
iqa-monitoring
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

Airflow reste en mode leger :
```text
LocalExecutor
PostgreSQL metadata DB
pool GPU Airflow max_active_tasks=1
pas de CeleryExecutor
pas de KubernetesExecutor
concurrence limitee
```

## 9. Tests

Tests attendus :
- import des DAGs Airflow : zero broken DAG en CI ;
- contrats API : `/health`, `/predict`, `/feedback`, `/model/version`, `/admin/reload-model` ;
- invariants datasets : aucun defaut, aucune ROI warning/fail, aucun validation_set dans le train ;
- gates : cas limites AP, taux Orange, FN ;
- storage : mapping URI logiques -> buckets/cles ;
- MinIO integration : round-trip ecriture/lecture ;
- model loading : chargement depuis `s3://iqa-models` ;
- feedback : oracle GT automatise le MVP ; Sophie reste une vitrine de revue ;
- aggregation piece : Rouge/Orange/Vert ;
- incidents rejouables : faux negatif, pic ROI fail, rollback.

## 10. Phases d'implementation

1. Squelette repo, configs, CI, PostgreSQL, MinIO, API `/health`.
2. Tracer bullet : une piece traverse predict -> feedback -> PostgreSQL -> MLflow.
3. Fondation donnees : inventory, piece_events, `validation_set_v001`, replay regenere.
4. Pipeline vision complet multi-vues.
5. Replay naturel via API reelle.
6. Dataset builder candidat good-only.
7. Evaluation et gates chiffrees.
8. Boucle modele Airflow.
9. Scenario drift + isolation/reset par `scenario_id`.
10. Monitoring Prometheus/Grafana.
11. Streamlit dashboard Marc + review Sophie.
12. Incidents rejouables, dont rollback via MLflow Registry.
13. Deploiement Z420 Ubuntu Server + runbook.

## 11. Decisions retenues

- Ubuntu Server est la cible officielle de deploiement ; Windows + WSL2 n'est pas retenu.
- Airflow est retenu pour la demonstration MLOps.
- La CI ne declenche jamais d'entrainement.
- Le lifecycle modele est declenche par evenement donnees.
- PostgreSQL est le metadata store cible.
- MinIO est le stockage objet local cible.
- DVC utilise le remote `s3://iqa-dvc`.
- Le module `src/iqa/storage` est le seul a parler a MinIO/boto3.
- Le validation set est fige avant replay et hors calibration.
- Le calibration set est good-only et exclu de bootstrap, replay, train et validation.
- Les scenarios sont isoles par `scenario_id`.
- L'interface Sophie est une vitrine MVP ; le feedback operationnel est l'oracle GT.
- MLflow Registry est la source de verite de la promotion et du rollback.
- Seul le Feature-AE est reentraine automatiquement.
- Kubernetes et reentrainement ROI sont hors MVP.

## 12. Verdict

Cette architecture devient la base cible IQA.

Elle combine :
```text
template MLOps + microservices Docker + Airflow + MinIO + validation set fige
+ oracle GT MVP + Feature-AE lifecycle + Ubuntu Server
```

Elle reste suffisamment simple pour le MVP, tout en donnant une colonne vertebrale credible pour la soutenance et le deploiement sur la Z420.

## 13. Decisions de convergence Ken/IQA

Decisions adoptees :

- `piece_event` est l'unite atomique de split, replay, feedback, validation,
  calibration et train.
- `calibration_set_v001` est good-only, fige avant replay, hors bootstrap,
  replay, train et `validation_set_v001`.
- Invariant dataset : `bootstrap ∩ calibration ∩ replay ∩ validation = vide`.
- `event_time` represente le temps simule du replay ; `recorded_at` represente
  l'horloge systeme ; `is_simulated` est derive de la source.
- MLflow Registry est la source de verite ; MinIO stocke les artefacts.
- Registered models par scenario : `feature_ae__production_replay_natural` et
  `feature_ae__drift_domain_extension`.
- API et inference restent deux services separes : `iqa-api` et `iqa-inference`.
- Le pyproject.toml racine est conserve en phase initiale ; la migration vers
  un dossier `services/` est reportee.
- Sophie reste une vitrine MVP ; `human_sophie` est futur, `oracle_gt` pilote le
  workflow operationnel.
