-- Provisions the logical databases expected by .env.example.
-- The default "iqa" user is created by the postgres image itself from
-- POSTGRES_USER / POSTGRES_PASSWORD and owns these databases.

CREATE DATABASE iqa_metadata OWNER iqa;
CREATE DATABASE mlflow OWNER iqa;
CREATE DATABASE airflow OWNER iqa;
