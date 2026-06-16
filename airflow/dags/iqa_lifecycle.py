"""IQA Feature-AE lifecycle DAG.

Pipeline stages:
  dataset → train → eval → gates → mlflow → promotion → reload
"""

from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:  # pragma: no cover
    DAG = None
    PythonOperator = None

try:
    from iqa.dags.lifecycle_tasks import (
        task_dataset,
        task_eval,
        task_gates,
        task_mlflow,
        task_promotion,
        task_reload,
        task_train,
    )
except ImportError:  # pragma: no cover
    def _placeholder_task(**_context):
        return {"status": "placeholder", "reason": "iqa package not available in airflow image"}

    task_dataset = _placeholder_task
    task_train = _placeholder_task
    task_eval = _placeholder_task
    task_gates = _placeholder_task
    task_mlflow = _placeholder_task
    task_promotion = _placeholder_task
    task_reload = _placeholder_task


GPU_POOL = "iqa_gpu"


dag = None
if (
    DAG is not None
    and PythonOperator is not None
    and all(
        [
            task_dataset,
            task_train,
            task_eval,
            task_gates,
            task_mlflow,
            task_promotion,
            task_reload,
        ]
    )
):
    with DAG(
        dag_id="iqa_lifecycle",
        schedule=None,
        catchup=False,
        start_date=datetime(2026, 1, 1),
        tags=["iqa", "lifecycle"],
        params={
            "regime": "natural",
            "scenario_id": "production_replay_natural",
        },
    ) as dag:
        op_dataset = PythonOperator(
            task_id="dataset",
            python_callable=task_dataset,
            doc="Prepare dataset for training",
        )

        op_train = PythonOperator(
            task_id="train",
            python_callable=task_train,
            pool=GPU_POOL,
            doc="Train model",
        )

        op_eval = PythonOperator(
            task_id="eval",
            python_callable=task_eval,
            pool=GPU_POOL,
            doc="Evaluate model",
        )

        op_gates = PythonOperator(
            task_id="gates",
            python_callable=task_gates,
            doc="Check promotion gates",
        )

        op_mlflow = PythonOperator(
            task_id="mlflow",
            python_callable=task_mlflow,
            doc="Register model in MLflow",
        )

        op_promotion = PythonOperator(
            task_id="promotion",
            python_callable=task_promotion,
            doc="Promote model to production",
        )

        op_reload = PythonOperator(
            task_id="reload",
            python_callable=task_reload,
            doc="Reload model in inference service",
        )

        # Linear dependencies: dataset → train → eval → gates → mlflow → promotion → reload
        op_dataset >> op_train >> op_eval >> op_gates >> op_mlflow >> op_promotion >> op_reload
