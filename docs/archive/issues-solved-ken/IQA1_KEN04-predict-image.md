# IQA1_KEN04 — predict_image minimal (MOD)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 100 % · **Dates :** 11/06 → 12/06

## What to build

Implémenter `predict_image` minimal (cf. `scripts/predict_image.py` / `iqa.inference.service`) qui produit, pour une image :

- `score`
- `ROI status`
- `heatmap` (placeholder accepté)
- `latency_ms`

## Acceptance criteria

- [ ] `predict_image` retourne `score`, `roi_status`, `heatmap` (placeholder), `latency_ms`
- [ ] La sortie respecte le contrat `IQA1_KEN01`
- [ ] `latency_ms` mesuré sur l'inférence
- [ ] Test d'inférence sur une image d'exemple

## Blocked by

- IQA1_KEN01 (contrat modèle)
- IQA1_KEN03 (wrappers)
