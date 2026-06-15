# IQA2_KEN08 — Gates de promotion (GATE)

**Type :** AFK · **Charge :** — · **Dates :** 16/06 → 17/06

## What to build

Implémenter les gates de promotion (config `configs/promotion_gates.yaml`) :

- `recall = 1.0` (aucun faux négatif)
- `AP prod -0.02` max (régression AP tolérée)
- `Orange rate` (seuil)
- `latency` (seuil)
- déclenchement `rollback` si gate échouée

## Acceptance criteria

- [ ] Gate recall = 1.0 / zéro FN bloquant
- [ ] Gate régression AP ≤ 0.02 vs prod
- [ ] Gate Orange rate
- [ ] Gate latency
- [ ] Échec de gate => signal de rollback
- [ ] Tests par gate (passant / bloquant)

## Blocked by

- IQA2_KEN05 (defect_coverage gate)
- IQA2_KEN07 (registered models)
