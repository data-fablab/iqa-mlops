# IQA2_KEN16 — Tests promotion / rollback (TST)

**Type :** AFK · **Charge :** — · **Dates :** 19/06 → 19/06

## What to build

Créer les tests du cycle promotion / rollback :

- FN bloque la promotion
- AP insuffisante bloque
- promotion success
- rollback restore

## Acceptance criteria

- [ ] Test : un faux négatif bloque la promotion
- [ ] Test : AP insuffisante (> -0.02) bloque
- [ ] Test : promotion réussie quand gates passées
- [ ] Test : rollback restaure `previous_prod`

## Blocked by

- IQA2_KEN09 (promotion)
- IQA2_KEN10 (rollback)
