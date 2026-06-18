# 13 - DAG iqa_monitoring reecrit en conteneur

Type : AFK

## What to build

Reecrire `airflow/dags/iqa_monitoring.py` via la factory : lancement de l'image
`data` avec `iqa-run-monitoring`. Les metriques continuent d'alimenter
Prometheus/Grafana (via statsd-exporter pour Airflow, et l'export propre au job).

## Acceptance criteria

- [x] `iqa_monitoring` n'utilise plus d'operateur important `iqa`
  (BashOperator -> `make_container_task`, plus aucune reference au code metier)
- [x] Lancement conteneur image `data` avec `iqa-run-monitoring`
  (argv templatise ; `drift_confirmed` passe en valeur pour un argv statique)
- [x] Les seuils `configs/monitoring_thresholds.yaml` sont evalues dans le conteneur
  (`iqa-run-monitoring` charge le YAML et compare `roi_fail_rate` aux seuils
  warning/critical ; verifie : roi 0.12 -> `critical`/breached)
- [~] Metriques visibles dans Grafana apres un run (frontiere validated-summary :
  l'evaluation des seuils est reelle ; le push Prometheus/Grafana est runtime
  observabilite, isole dans l'issue 23)
- [x] Import DagBag vert

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)

## Sibling (runtime)

- 23 - Runtime monitoring : export des metriques vers Prometheus / Grafana
