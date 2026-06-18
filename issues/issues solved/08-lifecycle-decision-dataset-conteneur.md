# 08 - Lifecycle (1/4) : lifecycle_decision + dataset en conteneurs

Type : AFK

## What to build

Premiere tranche de la reecriture de `iqa_lifecycle.py` : remplacer les
`PythonOperator` important `iqa.dags.lifecycle_tasks` par des taches conteneur via
la factory pour `lifecycle_decision` et `dataset`. Resout le debut de l'incoherence
ADR 0008 (Airflow n'importe plus `iqa`). Les params du DAG (regime, scenario_id,
seuils) restent passes en variables d'env du conteneur.

## Acceptance criteria

- [x] `lifecycle_decision` et `dataset` s'executent en conteneurs (image data)
      via `make_container_task` (`iqa-run-lifecycle-decision`, `iqa-run-dataset`).
- [x] Plus aucun `import iqa` ni placeholder sur ces deux taches (le scheduler ne
      les route plus via `iqa.dags.lifecycle_tasks`).
- [x] La decision de declenchement par evenement (drift/lot/volume) est evaluee dans
      le conteneur (`iqa-run-lifecycle-decision` appelle `evaluate_lifecycle_signal`).
- [~] Le dataset prepare est materialise (MinIO/PostgreSQL) et lisible par l'etape
      suivante. **Differe** : `iqa-run-dataset` est une frontiere "validated-summary"
      (`materialized: false`). La materialisation reelle est isolee dans l'**issue 19**
      (meme decoupage que 07->18 ; cf. cadrage `issues/README.md`).
- [x] Import DagBag vert (DAG hybride : 2 taches conteneur + 6 PythonOperator ;
      DagBag KO sans provider Docker -> `dag=None` garde, tests `docker_contract` skip).

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)

## Suivi

- Persistance runtime du dataset : **issue 19** (debloquee par celle-ci).
- Conteneurisation du reste du lifecycle : issues 09 (train/eval), 10 (gates/mlflow),
  11 (promotion/reload).
