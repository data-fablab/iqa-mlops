# IQA1_KEN08 — Registry skeleton par scenario_id (REG)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 50 % · **Dates :** 12/06 → 12/06

## What to build

Créer le squelette du registry (`iqa.registry.mlflow_registry`) gérant les états du cycle de vie, par `scenario_id` :

- `candidate`
- `test`
- `prod`
- `archived`

## Acceptance criteria

- [ ] Les quatre états sont modélisés
- [ ] Le registry est partitionné par `scenario_id`
- [ ] API pour lister / récupérer le modèle d'un état pour un `scenario_id`
- [ ] Test de contrat registry (`tests/test_registry_contract.py`)

## Blocked by

- IQA1_KEN07 (MLflow source de vérité)
