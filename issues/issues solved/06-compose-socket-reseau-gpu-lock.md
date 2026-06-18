# 06 - Cablage compose : socket Docker, reseau partage, lock GPU

Type : AFK

## What to build

Permettre au scheduler Airflow de lancer des conteneurs de tache : monter
`/var/run/docker.sock`, partager le reseau pour atteindre postgres/minio/mlflow/
inference, et propager le lock GPU (volume `gpu_lock`) aux conteneurs de tache GPU.

## Acceptance criteria

- [x] Socket Docker monte sur le scheduler (`/var/run/docker.sock` + `group_add: IQA_DOCKER_GID` ; `docker ps` OK depuis le scheduler)
- [x] Les conteneurs de tache rejoignent le reseau des services et resolvent leurs noms (reseau fixe `iqa_net` ; `IQA_DOCKER_NETWORK=iqa_net` consomme par la factory)
- [x] Volume `gpu_lock` propage aux taches GPU ; un seul detenteur a la fois (volume nomme `iqa_gpu_lock` + `make_container_task(gpu_lock=True)` monte le lock partage avec l'inference)
- [x] La tache tracer de l'issue 05 tourne reellement via compose (run `tracer_smoke_1` -> tache `run_container` = **success** ; conteneur `iqa-data:local` lance par le scheduler, exit code remonte dans Airflow)

## Blocked by

- 05 - Factory make_container_task (docker|k8s)
