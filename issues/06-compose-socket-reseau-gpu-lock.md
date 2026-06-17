# 06 - Cablage compose : socket Docker, reseau partage, lock GPU

Type : AFK

## What to build

Permettre au scheduler Airflow de lancer des conteneurs de tache : monter
`/var/run/docker.sock`, partager le reseau pour atteindre postgres/minio/mlflow/
inference, et propager le lock GPU (volume `gpu_lock`) aux conteneurs de tache GPU.

## Acceptance criteria

- [ ] Socket Docker monte sur le scheduler
- [ ] Les conteneurs de tache rejoignent le reseau des services et resolvent leurs noms
- [ ] Volume `gpu_lock` propage aux taches GPU ; un seul detenteur a la fois
- [ ] La tache tracer de l'issue 05 tourne reellement via compose

## Blocked by

- 05 - Factory make_container_task (docker|k8s)
