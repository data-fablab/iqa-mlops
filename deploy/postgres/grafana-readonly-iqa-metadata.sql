-- Read-only Grafana role grants for the IQA metadata DB (predictions, feedback...).
--
-- Companion to grafana-readonly.sql (which grants on the mlflow DB). The Grafana
-- "IQA Metadata" Postgres datasource connects as grafana_ro to browse persisted
-- predictions/feedback. SELECT-only: Grafana can never write to iqa_metadata.
--
-- The grafana_ro role is created by grafana-readonly.sql (run that first, or on the
-- mlflow DB). Schema (predictions, feedback_events, ...) is created by the API on
-- first write (iqa.metadata.repository.create_metadata_repository ->
-- initialize_metadata_db), so run this AFTER at least one /predict, or re-run it
-- (idempotent) once the tables exist.
--
--   docker exec -i deploy-postgres-1 psql -U iqa -d iqa_metadata -f - < deploy/postgres/grafana-readonly-iqa-metadata.sql

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_ro') THEN
    CREATE ROLE grafana_ro LOGIN PASSWORD 'grafana_ro_password';
  END IF;
END
$$;

GRANT CONNECT ON DATABASE iqa_metadata TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_ro;

-- Future tables created by the owner (iqa) stay readable without re-running grants.
ALTER DEFAULT PRIVILEGES FOR ROLE iqa IN SCHEMA public GRANT SELECT ON TABLES TO grafana_ro;
