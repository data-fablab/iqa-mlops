# IQA1_KEN07 — MLflow Registry source de vérité + MinIO artefacts (MLF)

**Type :** HITL · **Charge :** 0,25 j · **Avancement initial :** 100 % · **Dates :** 12/06 → 12/06

## What to build

Acter formellement que **MLflow Registry est la source de vérité** des modèles et que **MinIO stocke les artefacts** (déjà partiellement couvert par ADR 0006 et ADR 0003). Vérifier la cohérence config / ADR.

## Acceptance criteria

- [ ] ADR 0006 (MLflow source de vérité) confirmé / à jour
- [ ] ADR 0003 (MinIO stockage objet) confirmé / à jour
- [ ] Config (`configs/paths.yaml`, `.env.example`) cohérente avec la décision
- [ ] Décision référencée depuis la doc registry

## Blocked by

None - can start immediately
