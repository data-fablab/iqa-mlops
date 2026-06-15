# IQA2_KEN15 — Tests ML sécurité IA (TST)

**Type :** AFK · **Charge :** — · **Dates :** 19/06 → 19/06

## What to build

Créer les tests ML garantissant les règles de sécurité IA et le blocage des gates :

- `no defective train`
- `no validation_set train`
- ROI `fail` exclus
- gates bloquantes

## Acceptance criteria

- [ ] Test : aucune pièce défectueuse dans le train
- [ ] Test : aucune donnée `validation_set` dans le train
- [ ] Test : ROI `fail` exclus du train
- [ ] Test : gates bloquantes empêchent la promotion
- [ ] Tests verts en CI

## Blocked by

- IQA2_KEN02 (candidate_builder good only)
- IQA2_KEN08 (gates)
