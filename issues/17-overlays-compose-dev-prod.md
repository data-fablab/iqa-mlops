# 17 - Overlays Compose dev / prod

Type : AFK

## What to build

Deux setups Docker a partir du compose de base, par overlays (pas un Dockerfile par
service : le multi-stage + extras de l'ADR 0007 suffit). Le prod/dev se differencie
au niveau Compose :

- `deploy/docker-compose.dev.yml` : bind-mount `../src` + `../scripts`, hot-reload
  (`iqa-api --reload`, `iqa-inference --reload`), build local.
- `deploy/docker-compose.prod.yml` : `restart: unless-stopped`, pas de bind-mount,
  images par role tirees du registre via `${IQA_IMAGE_REGISTRY}`/`${IQA_IMAGE_TAG}`
  (`build:` du compose de base sert de repli tant que le registre n'est pas branche).

Usage :
`docker compose -f docker-compose.yml -f docker-compose.dev.yml up`
`docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d`

## Acceptance criteria

- [x] Flag `--reload` ajoute a `iqa-api` / `iqa-inference` (scripts run_api / run_inference)
- [x] Overlay dev : code live monte par-dessus `/app/src` (PYTHONPATH), hot-reload
- [x] Overlay prod : `restart: unless-stopped` + images par role taggees du registre
- [x] `docker compose config` valide pour les deux combinaisons
- [ ] Une fois le registre branche (issue 15) : prod tire les tags sans `build:`

## Blocked by

- 02 - Dockerfile multi-stage + image iqa-api slim (overlays s'appuient sur les targets)

## Note

Converge avec l'issue 15 (images du registre par tag) cote prod : memes variables
`IQA_IMAGE_REGISTRY` / `IQA_IMAGE_TAG`.
