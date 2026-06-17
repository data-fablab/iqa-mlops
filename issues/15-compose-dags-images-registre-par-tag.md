# 15 - Compose + DAGs referencent les images du registre par tag

Type : AFK

## What to build

Basculer du `build: context: ..` local vers la consommation des images publiees sur
Docker Hub, referencees par tag, dans `deploy/docker-compose.yml` et dans les appels
de la factory cote DAGs. Le tag par defaut est parametrable (env).

## Acceptance criteria

- [ ] Le compose utilise `image: <org>/iqa-*:<tag>` au lieu de `build:` pour les services iqa
- [ ] La factory `make_container_task` reference les images du registre par tag
- [ ] Tag parametrable via variable d'environnement (defaut documente)
- [ ] `docker compose up` fonctionne sans build local (images pull depuis le registre)
- [ ] Smoke-tests verts sur images tirees du registre

## Blocked by

- 14 - CI : build + push des images sur Docker Hub
