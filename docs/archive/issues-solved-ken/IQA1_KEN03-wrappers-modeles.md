# IQA1_KEN03 — Wrappers modèles (MOD)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 100 % · **Dates :** 11/06 → 11/06

## What to build

Créer les wrappers d'inférence pour les trois modèles, avec interface homogène (`load` / `predict`) :

- ROI segmenter (`iqa.models.segmentation`)
- Teacher ResNet18 (`iqa.models.feature_ae.teacher`)
- Feature AE (`iqa.models.feature_ae`)

## Acceptance criteria

- [ ] Wrapper ROI segmenter expose `load` + `predict`
- [ ] Wrapper Teacher ResNet18 expose `load` + `predict`
- [ ] Wrapper Feature AE expose `load` + `predict`
- [ ] Interface homogène entre les trois wrappers
- [ ] Test d'import / smoke test des wrappers

## Blocked by

- IQA1_KEN01 (contrat modèle)
