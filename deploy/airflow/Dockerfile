# Airflow image for IQA: base Airflow + the Docker provider so the scheduler can
# launch task containers via DockerOperator (ADR 0008, issue 06). The iqa package
# is NOT installed here -- only `iqa.dags.operators` is needed and it is mounted
# read-only on PYTHONPATH (it imports no metier runtime).
FROM apache/airflow:2.10.5-python3.12

# apache-airflow-providers-docker pulls the docker SDK (docker.types.Mount).
RUN pip install --no-cache-dir "apache-airflow-providers-docker>=3.12,<4"
