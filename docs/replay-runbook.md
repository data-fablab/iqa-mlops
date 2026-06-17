# Replay API Runbook

Le replay Phase 2 expose des runs API stables au-dessus des manifests CSV. Le
backend actuel est file-backed ; l'interface est compatible avec une persistance
PostgreSQL future.

## API

```bash
curl http://localhost:8000/replay-scenarios
curl -X POST http://localhost:8000/replay-runs \
  -H "Content-Type: application/json" \
  -d '{"scenario_id":"production_replay_natural"}'
curl http://localhost:8000/replay-runs/<replay_run_id>/next
curl -X POST http://localhost:8000/replay-runs/<replay_run_id>/reset
```

Scenarios supportes :

- `production_replay_natural`
- `drift_domain_extension`

Chaque run conserve son curseur. Un reset ne modifie pas les autres runs.
Deux runs du meme scenario servent les evenements dans le meme ordre stable.
Les reponses de run exposent aussi `lot_ids`, `source_classes`, `created_at` et
`updated_at`. Chaque evenement servi porte `replay_run_id`, `replay_position`,
`served_at`, `lot_id`, `source_class`, `scheduled_at` et `event_time`.

Le scheduler file-backed utilise les manifests CSV comme source d'evenements.
Les scenarios restent declares dans `src/iqa/replay/scenarios.py` pour ce lot ;
un `replay_scenarios.csv` pourra etre materialise plus tard si Airflow ou DVC en
ont besoin comme artefact.

`production_replay_natural` suit l'ordre stable du manifest. Le scenario
`drift_domain_extension` est servi par regime de source, dans l'ordre
`Casting_class1 -> Casting_class2 -> Casting_class3`, tout en conservant l'ordre
stable interne de chaque classe.

## Airflow Docker

Verification des DAGs sur serveur :

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec airflow-webserver airflow dags list
docker compose --env-file .env -f deploy/docker-compose.yml exec airflow-webserver airflow dags list-import-errors
```

Les DAGs doivent etre relancables sans ecriture partielle silencieuse. Les taches
doivent loguer les chemins de manifests, `scenario_id`, `dataset_version` et le
statut d'acceptation ou de refus.
