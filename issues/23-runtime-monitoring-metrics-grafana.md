# 23 - Runtime monitoring : export des metriques vers Prometheus / Grafana

Type : AFK

## What to build

Implementer l'export reel des metriques de monitoring vers Prometheus/Grafana,
aujourd'hui absent : `scripts/run_monitoring.py` (`iqa-run-monitoring`, issue 13)
evalue le signal lifecycle **et** les seuils (`configs/monitoring_thresholds.yaml`)
dans le conteneur (reel), et imprime un resume JSON, **sans pousser** les metriques
a Prometheus/Grafana. L'issue 13 a conteneurise la tache `evaluate_lifecycle_conditions`
du DAG ; celle-ci comble le runtime observabilite.

La stack expose deja `statsd-exporter` (metriques Airflow), `prometheus` et
`grafana` (cf. `deploy/docker-compose.yml`). Cabler l'export propre au job : le
conteneur de monitoring pousse `roi_fail_rate`, le statut de seuil (ok/warning/
critical) et `trigger_lifecycle` vers Prometheus (push gateway ou statsd), visibles
dans un dashboard Grafana.

## Acceptance criteria

- [ ] `iqa-run-monitoring` exporte ses metriques (roi_fail_rate, statut de seuil,
  trigger_lifecycle) vers Prometheus
- [ ] Les metriques sont visibles dans Grafana apres un run du DAG `iqa_monitoring`
- [ ] L'evaluation des seuils (deja reelle, issue 13) alimente les valeurs exportees
- [ ] Tests : couverture de l'export ; suite + DagBag verts

## Blocked by

- 13 - DAG iqa_monitoring reecrit en conteneur

## Note

Decoupage coherent avec 07->18, 08->19, 09->20, 10->21, 11->22 : la conversion DAG
(legere ; l'evaluation des seuils dans le conteneur est deja reelle) et le runtime
observabilite (push Prometheus + dashboard Grafana) sont deux travaux distincts
(cf. cadrage `issues/README.md`).
