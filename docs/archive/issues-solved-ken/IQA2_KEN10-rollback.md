# IQA2_KEN10 — Rollback (REG)

**Type :** AFK · **Charge :** — · **Dates :** 17/06 → 17/06

## What to build

Implémenter le rollback via :

- transition d'état MLflow (retour de l'ancien modèle en `prod`)
- restauration de `previous_prod`

## Acceptance criteria

- [ ] `previous_prod` tracé avant chaque promotion
- [ ] Rollback restaure `previous_prod` en `prod` via transition MLflow
- [ ] Le modèle fautif est `archived`
- [ ] Test : rollback restaure l'état précédent

## Blocked by

- IQA2_KEN09 (promotion)
