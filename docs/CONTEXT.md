# CONTEXT

IQA est un MVP MLOps pour le controle visuel de pieces `Casting`.

Le dataset Casting sert d'historique rejoue (`historical_replay`). La production cible utilisera `production_ingest`. Dans les deux cas, les images sont stockees dans MinIO et les faits metier dans PostgreSQL.

Vocabulaire stockage :

- Dataset source : historique Casting immutable, stocke dans `s3://iqa-source-datasets` et/ou versionne DVC.
- Donnees ingerees : images passees par le contrat d'ingestion, stockees dans `s3://iqa-ingested-images`, avec `piece_event`, timestamps, source et URI dans PostgreSQL.
- Artefacts : sorties produites par les pipelines, par exemple heatmaps, modeles, runs MLflow, rapports et datasets candidats.

Regle : PostgreSQL stocke les faits et les URI ; MinIO stocke les fichiers lourds.

Les checkpoints PyTorch lourds sont stockes dans MinIO, principalement `s3://iqa-models`, et references par manifests Git.
