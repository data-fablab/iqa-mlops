# Roadmap IQA - Convergence Ken/IQA

Cette roadmap reprend les tranches verticales utiles de la proposition Ken, en
les adaptant a l'architecture IQA retenue : `pyproject.toml` racine conserve,
`iqa-api` et `iqa-inference` separes, oracle GT operationnel et Sophie vitrine
MVP.

## Phase 1 - Fondations et tracer bullet

- Socle Docker Compose : PostgreSQL trois bases, MinIO, MLflow, Nginx.
- `iqa-init` idempotent a definir comme job futur : V0 bootstrap reference dans
  les deux registered models MLflow.
- Ingestion : sha256, `piece_event`, `event_time`, `recorded_at`,
  `is_simulated`.
- API : `/health`, `/predict`, `/piece-events/{event_id}/predict`.
- Inference : placeholder puis runtime PyTorch separe.
- Aggregation piece : Vert / Orange / Rouge.

## Phase 2 - Feedback, datasets et gates

- `/feedback` accepte `oracle_gt` comme workflow MVP.
- `human_sophie` reste futur et non bloquant.
- Dataset builder : split au niveau `piece_event`, etancheite des sets.
- `calibration_set_v001` good-only, hors bootstrap/replay/train/validation.
- Gates minimales : FN bloquant, AP image/pixel, Orange%, latence.
- `defect_coverage` par `source_class` comme recevabilite ROI.

## Phase 3 - Registry, lifecycle et monitoring

- MLflow Registry source de verite.
- Registered models :
  `feature_ae__production_replay_natural` et
  `feature_ae__drift_domain_extension`.
- `/admin/reload-model` resout la version active via MLflow.
- DAGs Airflow : ingestion, replay, lifecycle, monitoring.
- Monitoring : drift features teacher, p95 reconstruction, ROI warning/fail,
  faux negatif.

## Phase 4 - Demonstration et durcissement

- Scenario naturel de bout en bout prioritaire.
- Scenario drift comme demonstration gouvernee : alerte, candidat, gates,
  promotion ou rejet.
- Lots d'incident hors train/calibration/replay nominal : FN, pic ROI fail,
  rollback.
- Grafana pour Marc ; Streamlit vitrine Sophie.
- Runbook Z420 Ubuntu Server.

## Reports explicites

- Migration complete vers `services/` avec un `pyproject.toml` par service.
- Feedback humain Sophie operationnel et prioritaire.
- Reentrainement du ROI segmenter.
- Kubernetes, OAuth/RBAC complet et dashboards avances non essentiels.
