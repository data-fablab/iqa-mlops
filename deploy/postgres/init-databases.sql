-- Provisions the additional databases/users expected by .env.example
-- (IQA_METADATA_DB_URL, IQA_MLFLOW_DB_URL, IQA_AIRFLOW_DB_URL).
-- The default "iqa" user/database are created by the postgres image itself
-- from POSTGRES_USER / POSTGRES_DB.

CREATE DATABASE iqa_metadata OWNER iqa;

CREATE USER mlflow WITH PASSWORD 'mlflow_password';
CREATE DATABASE mlflow OWNER mlflow;

CREATE USER airflow WITH PASSWORD 'airflow_password';
CREATE DATABASE airflow OWNER airflow;
