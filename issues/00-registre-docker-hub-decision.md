# 00 - Decision : registre Docker Hub, nommage/tags, secrets CI (HITL)

Type : HITL

## What to build

Figer la convention de publication des images de service avant tout build/push.
Decision humaine, pas de code : choix du compte/organisation Docker Hub, schema de
nommage des images (`<org>/iqa-api`, `<org>/iqa-ml`, `<org>/iqa-data` ou un depot
unique multi-tags), strategie de tags (`sha`, `latest`, `vX.Y`), et secrets CI a
provisionner (`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`).

## Acceptance criteria

- [ ] Compte/organisation Docker Hub choisi et accessible
- [ ] Convention de nommage des images documentee (un depot par service vs depot unique)
- [ ] Strategie de tags arretee (au minimum `sha` immuable + `latest`)
- [ ] Secrets CI listes et configures dans le repo GitHub
- [ ] Decision consignee (commentaire d'ADR 0008 ou note dans le README deploy)

## Blocked by

None - can start immediately
