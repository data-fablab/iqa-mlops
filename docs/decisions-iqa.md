# Decisions IQA

Ce document est l'index court des decisions actives. Les decisions detaillees
restent dans les ADR pour eviter les doublons.

## Decisions Actives

| Sujet | Source de verite |
| --- | --- |
| Validation set fige hors replay | [ADR 0001](adr/0001-validation-set-fige-hors-replay.md) |
| Airflow comme orchestrateur | [ADR 0002](adr/0002-airflow-comme-orchestrateur.md) |
| MinIO comme stockage objet local | [ADR 0003](adr/0003-minio-stockage-objet-local.md) |
| PostgreSQL comme metadata store | [ADR 0004](adr/0004-postgresql-comme-metadata-store.md) |
| Calibration set et split `piece_event` | [ADR 0005](adr/0005-calibration-set-etanche-et-split-piece-event.md) |
| MLflow Registry comme source de verite | [ADR 0006](adr/0006-mlflow-registry-source-verite.md) |
| Services Docker avec `pyproject.toml` racine | [ADR 0007](adr/0007-architecture-services-avec-pyproject-racine.md) |

## Synthese

- `piece_event` est l'unite atomique de split, replay, feedback, validation,
  calibration et training.
- PostgreSQL stocke les faits, statuts, timestamps, versions et URI.
- MinIO stocke les fichiers lourds : images, modeles, masques ROI, heatmaps et
  artefacts MLflow.
- MLflow Registry est la source de verite pour le modele actif, la promotion et
  le rollback.
- Le segmenteur ROI est fige pour le MVP ; le Feature-AE est le modele vivant.
- La CI code et le lifecycle modele Airflow restent deux boucles separees.

## Questions Encore Ouvertes

- Finaliser la strategie de persistance PostgreSQL pour predictions, feedbacks,
  incidents et model_versions.
- Stabiliser le workflow complet de promotion/rollback apres integration des
  modules lifecycle.
- Automatiser le replay et la generation des datasets candidats Phase 2.
