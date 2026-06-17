# 12 - DAG iqa_replay reecrit en conteneur

Type : AFK

## What to build

Reecrire `airflow/dags/iqa_replay.py` via la factory : la (les) tache(s) lancent
l'image `data` avec `iqa-run-replay`. Conserver la semantique des evenements rejoues
(`event_time`, `recorded_at`, `is_simulated`).

## Acceptance criteria

- [ ] `iqa_replay` n'utilise plus d'operateur important `iqa`
- [ ] Lancement conteneur image `data` avec `iqa-run-replay` et params de scenario
- [ ] Les evenements rejoues conservent `event_time`/`recorded_at`/`is_simulated`
- [ ] Import DagBag vert

## Blocked by

- 06 - Cablage compose : socket Docker, reseau, lock GPU
- 04 - Image data (ingestion, replay, monitoring)
