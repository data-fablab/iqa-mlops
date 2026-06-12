# ADR 0004 - PostgreSQL comme metadata store

## Statut

Accepte.

## Contexte

Le MVP IQA doit tracer les decisions qualite et les evenements MLOps de bout en bout : pieces inspectees, predictions API, feedbacks, lots, scenarios, drift, runs de pipeline, evaluations, promotions, archivage, rollback et reload modele.

Ces informations sont structurees, relationnelles et auditables. Elles ne doivent pas etre melangees avec les artefacts lourds comme les images, heatmaps, datasets candidats, checkpoints ou artefacts MLflow.

## Decision

Retenir PostgreSQL comme metadata store applicatif obligatoire du projet IQA.

PostgreSQL stocke :
- `piece_events` et lots inspectes ;
- source d'ingestion `historical_replay` ou `production_ingest` ;
- URI des images, heatmaps et artefacts associes ;
- predictions, scores, modeles utilises et latences ;
- feedbacks et labels valides apres prediction ;
- alertes drift, incidents et decisions de monitoring ;
- runs de lifecycle modele ;
- decisions de promotion, rollback et reload.

PostgreSQL est aussi la cible pour les backends metadata MLflow et Airflow.

## Consequences

- `.env.example` expose `IQA_METADATA_DB_URL=postgresql://...`.
- Les exemples Docker Compose doivent inclure un service `postgres`.
- PostgreSQL stocke les faits et les URI ; il ne stocke pas les images ni les fichiers lourds.
- Les documentations ne doivent pas presenter SQLite comme cible projet.
