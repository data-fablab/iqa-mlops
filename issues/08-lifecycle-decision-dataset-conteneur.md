# 08 - Lifecycle (1/4) : lifecycle_decision + dataset en conteneurs

Type : AFK

## What to build

Premiere tranche de la reecriture de `iqa_lifecycle.py` : remplacer les
`PythonOperator` important `iqa.dags.lifecycle_tasks` par des taches conteneur via
la factory pour `lifecycle_decision` et `dataset`. Resout le debut de l'incoherence
ADR 0008 (Airflow n'importe plus `iqa`). Les params du DAG (regime, scenario_id,
seuils) restent passes en variables d'env du conteneur.

## Acceptance criteria

- [ ] `lifecycle_decision` et `dataset` s'executent en conteneurs (image data)
- [ ] Plus aucun `import iqa` ni placeholder sur ces deux taches
- [ ] La decision de declenchement par evenement (drift/lot/volume) est evaluee dans le conteneur
- [ ] Le dataset prepare est materialise (MinIO/PostgreSQL) et lisible par l'etape suivante
- [ ] Import DagBag vert

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)
