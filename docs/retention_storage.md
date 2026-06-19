# Retention et stockage - IQA

Cartographie du stockage IQA (MinIO, DVC, PostgreSQL) et politiques de
retention. Reference d'exploitation pour `deploy_runbook.md`. Source
architecture : `architecture-iqa.md` (section stockage).

## 1. Principes

- MinIO est le stockage objet local cible (S3-compatible).
- DVC versionne les donnees lourdes avec remote `s3://iqa-dvc` ; les artefacts
  lourds sont exclus de Git.
- PostgreSQL est le metadata store (faits, URI, audit).
- Le module `src/iqa/storage` est le seul a parler a MinIO/boto3.
- MLflow Registry est la source de verite pour la promotion/rollback modele.

## 2. Buckets MinIO

Crees par `deploy/minio/init-buckets.sh` :

| Bucket | Contenu | Retention |
| --- | --- | --- |
| `iqa-source-datasets` | dataset historique immutable | conserver (immutable) |
| `iqa-dvc` | remote DVC (donnees, datasets candidats) | gere par DVC, conserver |
| `iqa-ingested-images` | images brutes recues ou rejouees | long terme (tracabilite) |
| `mlflow-artifacts` | artefacts MLflow (runs, modeles) | lie au cycle de vie des runs |
| `iqa-roi-masks` | masques ROI du segmenteur fige | long terme |
| `iqa-heatmaps` | heatmaps et overlays | mixte (voir section 3) |
| `iqa-gt-masks` | masques GT/oracle | long terme (preuve feedback) |
| `iqa-models` | modeles candidats, promus, archives | conserver (audit promotion) |
| `iqa-backups` | sauvegardes applicatives | rotation hors site |

## 3. Politique de retention heatmaps (ILM)

Definie dans `deploy/minio/lifecycle-heatmaps.json`, importee par
`mc ilm import` sur `iqa-heatmaps` :

- Prefixe `lots/` : heatmaps par lot, **expiration automatique a 30 jours**
  (regle `expire-heatmap-lots`). Volume eleve, valeur courte.
- Prefixe `curated/` : heatmaps revues conservees pour la demo / le rapport,
  **aucune expiration**.

Convention applicative : ecrire les heatmaps de production sous `lots/`, deplacer
sous `curated/` celles a conserver.

## 4. DVC

- Remote : `s3://iqa-dvc` (MinIO).
- Donnees versionnees : `data/raw/` (hss-iad), datasets candidats Feature-AE.
- Les checkpoints lourds (ex. bootstrap ROI) restent caches hors Git, references
  par DVC.
- Le validation set est fige avant replay ; le calibration set est good-only et
  exclu de bootstrap, replay, train et validation.

## 5. PostgreSQL (metadata store)

Un conteneur, trois bases logiques (provisionnees par
`deploy/postgres/init-databases.sql`) :

| Base | Usage | Retention |
| --- | --- | --- |
| `iqa_metadata` | faits piece_events, predictions, feedback, URI, audit | long terme |
| `mlflow` | backend store MLflow (runs, params, metrics, registry) | lie aux runs |
| `airflow` | metadata Airflow (DAG runs, tasks, pools) | rotation possible |

Note MVP : en Phase 1/2 l'historique des predictions de l'API est en memoire ;
la persistance PostgreSQL runtime est la cible.

## 6. Sauvegardes

- PostgreSQL : `pg_dumpall` planifie, depot dans `s3://iqa-backups`.
- MinIO : repliquer `s3://iqa-backups` (et buckets critiques) hors site.
- MLflow : couvert par la sauvegarde de la base `mlflow` + `mlflow-artifacts`.

## 7. Recapitulatif retention

```text
immutable / conserver : iqa-source-datasets, iqa-models, iqa-roi-masks, iqa-gt-masks
gere par outil         : iqa-dvc (DVC), mlflow-artifacts (MLflow)
expiration partielle   : iqa-heatmaps (lots/ = 30j, curated/ = garde)
long terme tracabilite : iqa-ingested-images, iqa_metadata (PostgreSQL)
rotation / hors site    : iqa-backups
```
