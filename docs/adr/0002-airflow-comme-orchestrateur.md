# ADR 0002 - Airflow comme orchestrateur

## Statut

Accepte.

## Contexte

Le projet doit demontrer une boucle MLOps lisible, avec DAGs, dependances, logs, retries et suivi des runs.

## Decision

Retenir Airflow comme orchestrateur MVP.

DAGs cibles :
- `iqa_ingestion.py` ;
- `iqa_replay.py` ;
- `iqa_lifecycle.py` ;
- `iqa_monitoring.py`.

Airflow fonctionne en mode leger :
```text
LocalExecutor
PostgreSQL metadata DB
pas de CeleryExecutor
pas de KubernetesExecutor
concurrence limitee
```

## Consequences

- Le DAG `iqa_lifecycle.py` devient la colonne vertebrale de la demonstration MLOps.
- Les entrainements ne sont jamais declenches par la CI.
- Les reentrainements sont declenches par evenement donnees : drift confirme, lot complet, ou volume suffisant de conformes valides.
