# 12 - DAG iqa_replay reecrit en conteneur

Type : AFK

## What to build

Reecrire `airflow/dags/iqa_replay.py` via la factory : la (les) tache(s) lancent
l'image `data` avec `iqa-run-replay`. Conserver la semantique des evenements rejoues
(`event_time`, `recorded_at`, `is_simulated`).

## Acceptance criteria

- [x] `iqa_replay` n'utilise plus d'operateur important `iqa`
  (BashOperator -> `make_container_task`, plus aucune reference au code metier)
- [x] Lancement conteneur image `data` avec `iqa-run-replay` et params de scenario
  (argv templatise : `--scenario-id`, `--plan` ; image `{{ params.image }}`)
- [x] Les evenements rejoues conservent `event_time`/`recorded_at`/`is_simulated`
  (`iqa-run-replay` valide le plan et reporte `preserved_event_fields` ; verifie sur
  le plan reel : 832 evenements, les 3 champs preserves). Frontiere validated-summary :
  l'emission reelle des evenements dans le store d'ingestion est runtime (issue 18).
- [x] Import DagBag vert (verifie en conteneur : DAG `iqa_replay` parse, 0 import error,
  tache `run_replay` = DockerOperator)

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)

## Note

Conversion DAG faite. La semantique des evenements rejoues est verifiable au niveau
frontiere (les 3 champs sont preserves car `iqa.replay` lit les lignes completes du
plan). L'emission reelle (rejouer les 832 evenements dans le pipeline d'ingestion en
conservant les timestamps) est du data plane : couverte par le runtime d'ingestion
(issue 18, puisque les evenements rejoues transitent par l'ingestion), pas une issue
soeur dediee.
