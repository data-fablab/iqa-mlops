# IQA2_KEN09 — Promotion MLflow source de vérité + MinIO artefacts (REG)

**Type :** AFK · **Charge :** — · **Dates :** 17/06 → 17/06

## What to build

Implémenter la promotion d'un modèle : transition d'état MLflow (`candidate` → `test`/`prod`) faisant foi, artefacts résolus depuis MinIO. Promotion conditionnée par les gates.

## Acceptance criteria

- [ ] Promotion = transition d'état MLflow (source de vérité)
- [ ] Artefacts du modèle promu résolus depuis MinIO
- [ ] Promotion bloquée si gates non passées
- [ ] Test de promotion réussie

## Blocked by

- IQA2_KEN08 (gates)
