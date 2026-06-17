# 05 - Factory make_container_task (docker|k8s)

Type : AFK

## What to build

Creer la factory qui encapsule le choix d'operateur Airflow, conformement a
l'ADR 0008. `make_container_task(task_id, image, command, env, pool, ...)` retourne
un `DockerOperator` aujourd'hui, parametrable vers `KubernetesPodOperator` via une
variable d'environnement (`IQA_AIRFLOW_BACKEND=docker|k8s`). Aucun import du runtime
`iqa` dans Airflow. Tracer : une tache de demonstration lance un conteneur.

## Acceptance criteria

- [ ] Module factory dans le package (ex. `src/iqa/dags/operators.py`) sans import du runtime metier
- [ ] `make_container_task` produit un `DockerOperator` parametre (image, command, env, pool)
- [ ] Bascule `IQA_AIRFLOW_BACKEND` prevue (chemin k8s stub/documente, non requis fonctionnel)
- [ ] Une tache tracer lance un conteneur et son code de sortie remonte dans Airflow
- [ ] Tests d'import des DAGs (DagBag) toujours verts

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim
