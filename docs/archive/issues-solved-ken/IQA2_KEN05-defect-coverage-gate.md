# IQA2_KEN05 — defect_coverage par source_class, blocage < 0.95 (ROI)

**Type :** AFK · **Charge :** — · **Dates :** 15/06 → 16/06

## What to build

Mesurer `defect_coverage` par `source_class` et **bloquer la recevabilité si couverture < 0.95** (implémentation du gate acté en IQA1_KEN02).

## Acceptance criteria

- [ ] Calcul de `defect_coverage` par `source_class`
- [ ] Gate bloquant si couverture < 0.95 (seuil depuis config)
- [ ] Résultat exposé au lifecycle / aux gates
- [ ] Test : couverture insuffisante => blocage

## Blocked by

- IQA1_KEN02 (gate defect_coverage acté)
- IQA2_KEN04 (evaluate)
