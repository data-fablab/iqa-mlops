# Roadmap IQA

Roadmap courte des travaux. Les details techniques vivent dans les documents
canoniques : architecture, PRD, runbook, ADR et docs modeles.

## Phase 1 - Fondations

- Socle Docker Compose : API, inference, PostgreSQL infra, MinIO, MLflow, Nginx.
- DVC et donnees historiques Casting disponibles via MinIO.
- ROI bootstrap operationnel avec checkpoint stocke hors Git.
- API, feedback MVP, metriques securite et gouvernance Phase 1 en place.
- Reste a stabiliser : lifecycle modele reel, DAGs au-dela des squelettes et
  persistance applicative PostgreSQL.

## Phase 2 - Boucle Realiste

- Brancher predictions, feedbacks, lots, incidents et model_versions sur
  PostgreSQL.
- Construire les datasets candidats depuis replay et feedback oracle.
- Entrainer, evaluer et logger les candidats Feature-AE dans MLflow.
- Appliquer les gates de promotion et rollback via MLflow Registry.
- Completer Airflow lifecycle, monitoring et runbooks d'exploitation.

## Phase 3 - Durcissement

- Automatiser les scenarios drift et incidents.
- Stabiliser dashboards Grafana et interface Streamlit.
- Ajouter retention MinIO, rapports de validation et controles de securite.
- Durcir l'acces equipe et les procedures de recuperation.

## Hors Scope MVP

- Kubernetes.
- OAuth/RBAC complet.
- Reentrainement automatique du segmenteur ROI.
- Migration multi-services avec un `pyproject.toml` par service.
