# Repartition Taches Phases 1-2

## Regle de vocabulaire commune

Le projet distingue trois familles de donnees. Cette separation doit rester visible dans les issues, les scripts, les DAGs, la documentation et les revues.

| Famille | Role | Stockage cible | Responsable technique |
|---|---|---|---|
| Dataset source | Historique Casting immutable utilise comme source de replay | `s3://iqa-source-datasets` et/ou DVC | Data / ingestion |
| Donnees ingerees | Images entrees dans le runtime via `historical_replay` ou `production_ingest` | images dans `s3://iqa-ingested-images`, faits et URI dans PostgreSQL | Ingestion / backend |
| Artefacts | Sorties produites par les pipelines : masques ROI, heatmaps, modeles, runs, rapports, datasets candidats | buckets dedies MinIO : `iqa-roi-masks`, `iqa-heatmaps`, `iqa-models`, `mlflow-artifacts`, `iqa-dvc` | ML / MLOps |

PostgreSQL ne stocke pas les images ni les checkpoints. Il stocke les faits, les statuts, les timestamps, les liens, les versions et les URI.

## Phase 1

- Restaurer structure repo, uv, tests, docs et contrats.
- Documenter dataset source vs donnees ingerees vs artefacts.
- Definir les buckets MinIO cibles : `iqa-source-datasets`, `iqa-ingested-images`, `iqa-dvc`, `mlflow-artifacts`, `iqa-roi-masks`, `iqa-heatmaps`, `iqa-models`, `iqa-backups`.
- Definir le contrat d'ingestion commun :
  - `historical_replay` pour le dataset Casting rejoue ;
  - `production_ingest` pour la future arrivee camera / poste qualite / MES.
- Stabiliser API skeleton, modeles runtime, storage et ingestion.
- Garder les CSV du repo comme manifests legers, pas comme stockage production.
- Verifier que les docs et tests de contrat utilisent le vocabulaire IQA.

Livrable Phase 1 :
```text
dataset source identifie
contrat ingestion defini
buckets documentes
PostgreSQL positionne comme metadata store
artefacts lourds exclus de Git
```

## Phase 2

- Brancher PostgreSQL, MinIO, MLflow et Airflow.
- Implementer le replay en passant par le contrat `historical_replay`, pas par un acces direct opportuniste aux fichiers sources.
- Implementer l'ecriture des images ingerees dans `s3://iqa-ingested-images`.
- Implementer l'ecriture des evenements, predictions, feedbacks et URI dans PostgreSQL.
- Implementer lifecycle Feature-AE, monitoring et promotion.
- Stocker les artefacts dans les buckets dedies :
  - masques ROI dans `s3://iqa-roi-masks` ;
  - heatmaps et overlays dans `s3://iqa-heatmaps` ;
  - modeles promus/candidats dans `s3://iqa-models` ;
  - runs et artefacts MLflow dans `s3://mlflow-artifacts` ;
  - datasets versionnes DVC dans `s3://iqa-dvc`.
- Ajouter interface Sophie/Marc.

Livrable Phase 2 :
```text
historical_replay -> ingestion -> MinIO/PostgreSQL -> inference
production_ingest pret a brancher sans changer le contrat
artefacts separes des donnees sources et des donnees ingerees
```
