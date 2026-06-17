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

## Airflow Docker

Verification des DAGs sur serveur :

```bash
docker compose --env-file .env -f deploy/docker-compose.yml exec airflow-webserver airflow dags list
docker compose --env-file .env -f deploy/docker-compose.yml exec airflow-webserver airflow dags list-import-errors
```

Les DAGs doivent etre relancables sans ecriture partielle silencieuse. Les taches
doivent loguer les chemins de manifests, `scenario_id`, `dataset_version` et le
statut d'acceptation ou de refus.
