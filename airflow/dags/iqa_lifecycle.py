"""IQA Feature-AE lifecycle DAG skeleton."""

from __future__ import annotations

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None


GPU_POOL = "iqa_gpu"

dag = None
if DAG is not None and BashOperator is not None:
    with DAG(
        dag_id="iqa_lifecycle",
        schedule=None,
        catchup=False,
        tags=["iqa", "lifecycle"],
    ) as dag:
        BashOperator(
            task_id="train_eval_gate_promote",
            bash_command="iqa-run-lifecycle",
            pool=GPU_POOL,
        )
