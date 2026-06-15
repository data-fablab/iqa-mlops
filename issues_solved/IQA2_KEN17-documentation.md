# IQA2_KEN17 — Documentation lifecycle (DOC)

**Type :** AFK · **Charge :** — · **Dates :** 19/06 → 19/06

## What to build

Documenter le système dans `docs/` :

- `model_lifecycle.md`
- `gates.md`
- `mlflow_registry.md`
- `drift_regimes.md`
- `rollback.md`

## Acceptance criteria

- [ ] `model_lifecycle.md` (dataset → reload)
- [ ] `gates.md` (recall, AP, Orange rate, latency, defect_coverage)
- [ ] `mlflow_registry.md` (états, scenario_id, source de vérité)
- [ ] `drift_regimes.md` (naturel / drift, baseline)
- [ ] `rollback.md` (transition, previous_prod)
- [ ] Docs cohérentes avec le code et les ADR

## Blocked by

- IQA2_KEN11 (DAG lifecycle complet)
- IQA2_KEN12, KEN13, KEN14
