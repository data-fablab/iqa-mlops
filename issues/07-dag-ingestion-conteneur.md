# 07 - DAG iqa_ingestion reecrit en conteneur

Type : AFK

## What to build

Reecrire `airflow/dags/iqa_ingestion.py` via la factory : la tache `run_ingestion`
lance l'image `data` avec la commande `iqa-run-ingestion`, au lieu du `BashOperator`
qui supposait `iqa` present dans l'image Airflow. DAG le plus simple, sert de
premier bout-en-bout reel.

## Acceptance criteria

- [ ] `iqa_ingestion` n'utilise plus `BashOperator`/CLI locale a Airflow
- [ ] La tache lance un conteneur image `data` avec les params (manifest, source, scenario_id)
- [ ] Un run complet ingere reellement (events en PostgreSQL, images en MinIO)
- [ ] Import DagBag vert

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)
