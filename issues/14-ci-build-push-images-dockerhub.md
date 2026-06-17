# 14 - CI : build + push des images par service sur Docker Hub (matrix)

Type : AFK

## What to build

Workflow GitHub Actions qui construit et pousse les images de service (serving, ml,
data) sur Docker Hub via une matrix, sur merge. Tags selon la convention de
l'issue 00 (`sha` immuable + `latest`). N'entraine jamais de modele (ADR 0002).

## Acceptance criteria

- [ ] Workflow CI avec matrix une entree par image (serving/ml/data)
- [ ] Build + push declenches sur merge vers la branche cible
- [ ] Images taggees `sha` + `latest` selon la convention de l'issue 00
- [ ] Secrets Docker Hub utilises depuis les secrets GitHub
- [ ] Aucun entrainement declenche par la CI

## Blocked by

- 00 - Decision registre Docker Hub
- 03 - Image ml (inference, trainer)
- 04 - Image data (ingestion, replay, monitoring)
