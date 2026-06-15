# IQA2_KEN13 — Baseline drift versionnée (MON)

**Type :** AFK · **Charge :** — · **Dates :** 18/06 → 18/06

## What to build

Créer une baseline de drift versionnée distinguant :

- `production inattendue` (drift subi)
- `extension domaine planifiée` (drift voulu)

Utilisée par le monitoring (`iqa.monitoring`) pour qualifier le drift.

## Acceptance criteria

- [ ] Baseline versionnée et stockée comme artefact
- [ ] Distinction `production inattendue` vs `extension domaine planifiée`
- [ ] Branchée au monitoring (`configs/monitoring_thresholds.yaml`)
- [ ] Test sur la qualification de drift

## Blocked by

- IQA2_KEN04 (evaluate)
