# 05 - Factory make_container_task (docker|k8s)

Type : AFK

## What to build

Creer la factory qui encapsule le choix d'operateur Airflow, conformement a
l'ADR 0008. `make_container_task(task_id, image, command, env, pool, ...)` retourne
un `DockerOperator` aujourd'hui, parametrable vers `KubernetesPodOperator` via une
variable d'environnement (`IQA_AIRFLOW_BACKEND=docker|k8s`). Aucun import du runtime
`iqa` dans Airflow. Tracer : une tache de demonstration lance un conteneur.

## Acceptance criteria

- [x] Module factory dans le package (`src/iqa/dags/operators.py`) sans import du runtime metier (test garde-fou inclus)
- [x] `make_container_task` produit un `DockerOperator` parametre (image, command, env, pool ; + `network_mode`/`docker_url` via env)
- [x] Bascule `IQA_AIRFLOW_BACKEND` prevue (docker par defaut ; chemin k8s `KubernetesPodOperator` stub documente, non requis fonctionnel)
- [x] Tache tracer `airflow/dags/iqa_container_tracer.py` cablee via la factory (1 tache = 1 conteneur, `command --help` exit 0) ; remontee du code de sortie reelle via compose = critere de l'issue 06
- [x] Tests d'import des DAGs (DagBag) toujours verts (suite : 476 passed, 13 skipped ; tracer import -> `dag=None` quand le provider Docker est absent en CI)

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim
