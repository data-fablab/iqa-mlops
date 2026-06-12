# PRD - IQA MVP

## 1. Probleme

Le controle visuel des pieces `Casting` depend de la fatigue, de l'eclairage et de la subjectivite de l'inspection humaine. Les defauts sont tres petits, parfois autour de 0,3 % a 0,6 % de l'image, et les decisions qualite ne sont pas reliees de facon exploitable aux images, modeles, versions de donnees et feedbacks.

Le MVP doit demontrer une boucle MLOps complete, gouvernee et tracable, sans remplacer Sophie.

## 2. Solution

IQA produit, pour chaque `piece_event` multi-vues :
- une decision Vert / Orange / Rouge ;
- un score d'anomalie ;
- une heatmap explicable ;
- une trace `sha256 -> piece_event -> scenario -> lot -> dataset_version -> model_version -> prediction -> feedback`.

Le dataset Casting est consomme comme un flux historique rejoue (`historical_replay`). La production reelle utilisera le meme contrat via `production_ingest`. Dans les deux cas, PostgreSQL conserve les faits et URI, tandis que MinIO conserve les fichiers images et artefacts.

Pipeline ML :
```text
image
-> ROI segmenter fige
-> controle qualite ROI
-> teacher ResNet18 fige
-> Feature-AE good-only
-> score + heatmap
-> aggregation multi-vues
-> feedback oracle GT puis human_sophie
```

## 3. Decisions produit

- Sophie reste decisionnaire.
- L'oracle GT accelere le MVP, mais le feedback humain prime.
- Le ROI segmenter et le teacher ResNet18 restent figes.
- Le Feature-AE est le seul modele vivant.
- Les scenarios de replay sont isoles par `scenario_id`.
- `validation_set_v001` est fige hors replay et hors calibration.
- Les promotions de modeles passent par MLflow et des gates chiffrees.
- La CI code et la boucle modele Airflow sont separees.
- Le serveur cible est Ubuntu Server sur Z420, pas Windows + WSL2.

## 4. User stories principales

1. Sophie veut voir une decision Vert / Orange / Rouge avec score et heatmap pour concentrer sa revue.
2. Sophie veut saisir un verdict humain pour corriger ou confirmer le systeme.
3. Marc veut suivre par lot les volumes, le taux Orange, les Rouges et le temps de controle.
4. Laurent veut auditer chaque prediction de bout en bout.
5. L'ingenieur MLOps veut rejouer deux scenarios : production naturelle et drift controle.
6. L'ingenieur MLOps veut declencher un reentrainement Feature-AE sur evenement donnees, pas sur commit.
7. L'equipe veut pouvoir promouvoir, archiver ou rollback un modele via MLflow.

## 5. Gates de promotion

Un candidat Feature-AE est promu seulement si :
```text
recall defaut == 1.0 sur validation_set_v001
Image AP et Pixel AP >= prod - 0.02
taux Orange <= 30 % en V0/V1
taux Orange <= 15 % en V2+
latence p95 sous objectif
incidents FN / ROI / rollback OK
```

Un faux negatif bloque toujours la promotion.

## 6. Scope MVP

Inclus :
- FastAPI ;
- Streamlit Sophie/Marc ;
- Airflow ;
- PostgreSQL ;
- MLflow ;
- DVC ;
- MinIO ;
- Prometheus/Grafana ;
- Feature-AE lifecycle ;
- replay naturel et drift controle ;
- incidents rejouables.

Hors scope :
- Kubernetes ;
- rejet automatique sans humain ;
- reentrainement ROI ;
- classification fine des defauts ;
- IoT machine ;
- stockage cloud externe ;
- Windows + WSL2 comme cible serveur officielle.

## 7. Jalons

```text
J7  -> tracer bullet : une piece traverse API, feedback, PostgreSQL, MLflow
J14 -> replay naturel + premiere boucle de promotion
J21 -> drift, dashboards, review Sophie, incidents
J24 -> feature freeze + deploiement Z420 Ubuntu + runbook
J28 -> soutenance
```
