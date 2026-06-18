# 07 - DAG iqa_ingestion reecrit en conteneur

Type : AFK

## What to build

Reecrire `airflow/dags/iqa_ingestion.py` via la factory : la tache `run_ingestion`
lance l'image `data` avec la commande `iqa-run-ingestion`, au lieu du `BashOperator`
qui supposait `iqa` present dans l'image Airflow. DAG le plus simple, sert de
premier bout-en-bout reel.

## Acceptance criteria

- [x] `iqa_ingestion` n'utilise plus `BashOperator`/CLI locale a Airflow (reecrit via `make_container_task`)
- [x] La tache lance un conteneur image `data` avec les params (manifest, source, scenario_id passes en argv templatises)
- [~] Un run complet ingere reellement : le conteneur execute la frontiere `iqa-run-ingestion` contre le manifeste reel (962 lignes, exit 0). **Persistance PG/MinIO non couverte** : `iqa-run-ingestion` est aujourd'hui une frontiere "validated-summary" (lit le CSV, n'ecrit ni events PG ni images MinIO). A realiser quand le runtime d'ingestion ecrira dans les stores (hors scope issue 07 ; respecter le contrat lineage : entree via MinIO).
- [x] Import DagBag vert (suite airflow : 32 passed, 8 skipped ; `dag=None` si provider Docker absent)

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)
