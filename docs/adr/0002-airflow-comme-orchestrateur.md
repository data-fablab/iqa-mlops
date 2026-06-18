# ADR 0002 - Airflow comme orchestrateur

## Statut

Accepte. Amende le 2026-06-17 (executor et evolution Kubernetes), voir [ADR 0008](0008-taches-airflow-comme-conteneurs.md).

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
concurrence limitee (un seul entrainement a la fois via le pool iqa_gpu)
```

Les taches s'executent comme conteneurs (voir [ADR 0008](0008-taches-airflow-comme-conteneurs.md)) :
- en MVP, `LocalExecutor` + `DockerOperator` ;
- l'evolution prevue de fin de projet est `KubernetesPodOperator` (et, si le temps le permet, `KubernetesExecutor`). Cette evolution n'est plus exclue : elle est planifiee comme cible optionnelle.

La portabilite Docker -> Kubernetes est garantie par la discipline "1 tache = 1 conteneur" : seul l'operateur change, pas les DAGs ni le runtime `iqa`.

## Consequences

- Le DAG `iqa_lifecycle.py` devient la colonne vertebrale de la demonstration MLOps.
- Les entrainements ne sont jamais declenches par la CI.
- Les reentrainements sont declenches par evenement donnees : drift confirme, lot complet, ou volume suffisant de conformes valides.
- Le passage a Kubernetes en fin de projet est une bascule d'operateur/executor, pas une refonte. Le pool de concurrence GPU devient une contrainte de ressource/affinite de noeud cote K8s.
