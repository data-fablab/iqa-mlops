# 13 - DAG iqa_monitoring reecrit en conteneur

Type : AFK

## What to build

Reecrire `airflow/dags/iqa_monitoring.py` via la factory : lancement de l'image
`data` avec `iqa-run-monitoring`. Les metriques continuent d'alimenter
Prometheus/Grafana (via statsd-exporter pour Airflow, et l'export propre au job).

## Acceptance criteria

- [ ] `iqa_monitoring` n'utilise plus d'operateur important `iqa`
- [ ] Lancement conteneur image `data` avec `iqa-run-monitoring`
- [ ] Les seuils `configs/monitoring_thresholds.yaml` sont evalues dans le conteneur
- [ ] Metriques visibles dans Grafana apres un run
- [ ] Import DagBag vert

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)
