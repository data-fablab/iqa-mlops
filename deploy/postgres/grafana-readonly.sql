-- Read-only Grafana role for the MLflow tracking DB (Issue 3).
--
-- Grafana's Postgres datasource connects as this role. It can only SELECT, so a
-- dashboard can never write to the mlflow backend store (acceptance criterion:
-- "aucune ecriture sur la base mlflow depuis Grafana").
--
-- Roles are cluster-wide but the SELECT grants need the mlflow schema to exist,
-- which mlflow creates on its first server start (Alembic migrations). So this is
-- applied AFTER mlflow is up, not from docker-entrypoint-initdb.d:
--
--   docker exec -i deploy-postgres-1 psql -U iqa -d mlflow -f - < deploy/postgres/grafana-readonly.sql
--
-- Idempotent: safe to re-run (e.g. after an mlflow upgrade adds tables).

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_ro') THEN
    CREATE ROLE grafana_ro LOGIN PASSWORD 'grafana_ro_password';
  END IF;
END
$$;

GRANT CONNECT ON DATABASE mlflow TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_ro;

-- Future tables created by the owner (iqa) stay readable without re-running grants.
ALTER DEFAULT PRIVILEGES FOR ROLE iqa IN SCHEMA public GRANT SELECT ON TABLES TO grafana_ro;
