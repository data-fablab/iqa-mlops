"""IQA DVC reproducibility gate DAG."""

from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None


dag = None
if DAG is not None and BashOperator is not None:
    with DAG(
        dag_id="iqa_dvc_reproducibility",
        schedule=None,
        catchup=False,
        start_date=datetime(2026, 1, 1),
        tags=["iqa", "dvc", "data-lineage"],
        params={
            "with_network": False,
            "skip_regeneration": False,
            "dvc_target": "data/raw/hss-iad.dvc",
        },
    ) as dag:
        BashOperator(
            task_id="dvc_reproducibility_check",
            bash_command=(
                "iqa-check-dvc-reproducibility "
                "{% if params.with_network %}--with-network {% endif %}"
                "{% if params.skip_regeneration %}--skip-regeneration {% endif %}"
            ),
        )
