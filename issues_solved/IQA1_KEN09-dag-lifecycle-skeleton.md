# IQA1_KEN09 — DAG IQA_lifecycle importable (AIR)

**Type :** AFK · **Charge :** 0,25 j · **Avancement initial :** 50 % · **Dates :** 12/06 → 12/06

## What to build

Créer le DAG `iqa_lifecycle` (`airflow/dags/iqa_lifecycle.py`) importable par Airflow, avec des tâches **placeholders** structurant le pipeline (dataset → train → eval → gates → mlflow → promotion → reload).

## Acceptance criteria

- [ ] DAG importable sans erreur (`DagBag` clean)
- [ ] Tâches placeholders pour chaque étape du lifecycle
- [ ] Dépendances entre tâches déclarées
- [ ] Test d'import du DAG

## Blocked by

None - can start immediately
