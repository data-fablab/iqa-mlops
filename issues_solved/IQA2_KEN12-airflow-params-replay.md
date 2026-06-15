# IQA2_KEN12 — Paramètres Airflow pour rejouer le lifecycle (AIR)

**Type :** AFK · **Charge :** — · **Dates :** 18/06 → 18/06

## What to build

Ajouter des paramètres Airflow (`params` / config) permettant de rejouer le lifecycle selon le régime, par `scenario_id` :

- régime `naturel`
- régime `drift`

## Acceptance criteria

- [ ] Paramètre de régime `naturel` / `drift`
- [ ] Paramètre `scenario_id`
- [ ] Le DAG rejoue le bon plan (`casting_flux_replay_plan_natural` / `_drift`)
- [ ] Test/validation d'un rejeu paramétré

## Blocked by

- IQA2_KEN11 (DAG lifecycle complet)
