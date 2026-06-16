# IQA1_KEN06 — Construction datasets candidats + règles sécurité IA (DATA)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 50 % · **Dates :** 12/06 → 12/06

## What to build

Créer le module de construction des datasets candidats (`candidate_builder`) appliquant les règles de sécurité IA :

- `good only`
- ROI OK uniquement
- aucun défaut
- exclusion du `validation_set`

## Acceptance criteria

- [ ] Module construit un dataset candidat versionné depuis les `piece_event`
- [ ] Filtre `good only` appliqué
- [ ] Filtre ROI OK appliqué
- [ ] Aucune pièce défectueuse incluse
- [ ] Aucune donnée du `validation_set` incluse
- [ ] Test sur les filtres de sécurité

## Blocked by

None - can start immediately
