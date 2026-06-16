# IQA1_KEN01 — Contrat modèle (MOD)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 100 % · **Dates :** 11/06 → 11/06

## What to build

Définir et figer le contrat d'entrée/sortie du modèle IQA, exposé via `src/iqa/inference/contracts.py`.

- **Entrée :** `piece_event`
- **Sorties :** `score`, `heatmap_uri`, `ROI status`, `statut`

Le contrat doit être typé (dataclass / pydantic), importable et couvert par un test de contrat.

## Acceptance criteria

- [ ] Le type d'entrée `piece_event` est défini et documenté
- [ ] La sortie expose `score`, `heatmap_uri`, `roi_status`, `statut`
- [ ] Le contrat est importable depuis `iqa.inference.contracts`
- [ ] Un test vérifie la forme du contrat (`tests/test_ml_source_contract.py` ou équivalent)

## Blocked by

None - can start immediately
