# ADR 0003 - MinIO comme stockage objet local

## Statut

Accepte.

## Contexte

Le projet doit stocker plusieurs familles d'artefacts : dataset source, images brutes ingerees, donnees DVC, artefacts MLflow, heatmaps, modeles promus, modeles precedents, candidats archives et backups.

## Decision

Retenir MinIO comme stockage objet S3-compatible local.

Buckets cibles :
```text
iqa-source-datasets
iqa-dvc
iqa-ingested-images
mlflow-artifacts
iqa-heatmaps
iqa-models
iqa-backups
```

Le code applicatif n'appelle pas directement boto3 partout. Un module unique encapsule l'acces :
```text
src/iqa/storage/
```

## Consequences

- Le dataset Casting source est conserve dans `s3://iqa-source-datasets`.
- DVC utilise `s3://iqa-dvc` avec `endpointurl` MinIO.
- Les images brutes recues en `production_ingest` ou rejouees en `historical_replay` sont stockees dans `s3://iqa-ingested-images` quand elles entrent dans le runtime.
- MLflow stocke ses artefacts dans `s3://mlflow-artifacts`.
- Les heatmaps sont stockees dans `s3://iqa-heatmaps`.
- Les modeles promus et candidats archives sont stockes dans `s3://iqa-models`.
- Les credentials MinIO restent dans `.env` et ne sont jamais commites.
