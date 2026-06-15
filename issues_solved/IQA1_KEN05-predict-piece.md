# IQA1_KEN05 — predict_piece minimal (MOD)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 100 % · **Dates :** 11/06 → 12/06

## What to build

Implémenter `predict_piece` minimal qui agrège les prédictions multi-vues d'une pièce et produit un statut métier :

- agrégation multi-vues (plusieurs `predict_image`)
- statut **Vert / Orange / Rouge**

## Acceptance criteria

- [ ] Agrégation de N vues d'une même pièce
- [ ] Règle d'agrégation produisant le statut Vert / Orange / Rouge
- [ ] Sortie conforme au contrat (`statut`)
- [ ] Test couvrant les trois statuts

## Blocked by

- IQA1_KEN04 (predict_image)
