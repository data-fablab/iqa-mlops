# Issue: Airflow DAG Tests Skip Without Airflow Installation

## Status
⚠️ **SKIPPED** — Not a failure, intentional design

## Description
6 tests in the Airflow DAG suite skip gracefully when Airflow is not installed:

```
tests/test_iqa_lifecycle_dag.py::test_dag_accepts_regime_and_scenario_id_params
tests/test_iqa_lifecycle_dag.py::test_dag_params_have_sensible_defaults
tests/test_iqa_lifecycle_dag.py::test_dag_can_run_with_drift_scenario
tests/test_airflow_dags.py::test_iqa_lifecycle_dag_has_seven_tasks
tests/test_airflow_dags.py::test_iqa_lifecycle_dag_has_linear_dependencies
tests/test_airflow_dags.py::test_iqa_lifecycle_dag_passes_dagbag_validation
```

## Root Cause
Airflow is an **optional dependency** for this project (heavy, used only for orchestration in production). The DAG code is wrapped in try/except:

```python
# airflow/dags/iqa_lifecycle.py
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None
    PythonOperator = None
```

When Airflow is not available, `DAG = None`, and the DAG creation is skipped (line 40-54).

## Skip Messages
- `DAG dependencies not available` → Airflow imports failed
- `DAG is None (Airflow not available)` → DAG creation was skipped  
- `Airflow not installed: No module named 'airflow.models'` → Direct import failure

## Resolution
**No action needed** — this is correct behavior.

### To test with Airflow (optional):
```bash
pip install apache-airflow  # Requires Python 3.9+, heavy dependencies
pytest tests/test_iqa_lifecycle_dag.py tests/test_airflow_dags.py -v
```

### Current CI/CD behavior:
- Essential tests **PASS**: DAG is importable, task structure is correct, source declares 7 tasks
- Execution tests **SKIP** gracefully: No Airflow → skip, not fail

## Tests that Always Pass (No Airflow Needed)
✅ `test_iqa_lifecycle_dag_imports_without_error`  
✅ `test_iqa_lifecycle_dag_source_declares_seven_tasks`

These tests verify the DAG **structure** without needing Airflow runtime.

## Related Files
- `airflow/dags/iqa_lifecycle.py` — DAG definition (safe imports)
- `src/iqa/dags/lifecycle_tasks.py` — Task implementations (all tested via mocks)
- `tests/test_iqa_lifecycle_dag.py` — DAG structure tests
- `tests/test_airflow_dags.py` — Airflow-specific tests
