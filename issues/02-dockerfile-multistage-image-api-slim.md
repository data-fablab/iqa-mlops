# 02 - Dockerfile multi-stage + image iqa-api slim (tracer)

Type : AFK

## What to build

Transformer le `Dockerfile` mono en multi-stage (un stage par extra de l'issue 01).
Tracer bullet : produire l'image `serving` pour `iqa-api`, sans PyTorch, et la
cabler dans `deploy/docker-compose.yml`. Le service doit demarrer et repondre.

C'est ce slice qui etablit le patron multi-stage reutilise par les images ml/data.

## Acceptance criteria

- [ ] `Dockerfile` multi-stage avec un stage par role (serving/ml/data)
- [ ] Image `iqa-api` construite depuis le stage `serving`, sans torch (taille reduite verifiee)
- [ ] `docker-compose.yml` build/utilise l'image serving pour `iqa-api`
- [ ] `GET /health` repond 200 sur le conteneur
- [ ] `deploy/smoke-test.sh` passe pour `iqa-api`

## Blocked by

- 01 - Decoupage des dependances en extras par role
