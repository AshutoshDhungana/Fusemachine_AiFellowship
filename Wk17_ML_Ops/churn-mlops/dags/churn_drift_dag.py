"""Airflow DAG (bonus): weekly drift check; on significant drift -> retrain (challenger vs champion)."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

REPO = Variable.get("churn_repo", default_var="/opt/churn-mlops")


def decide(**_):
    import json
    s = json.load(open(f"{REPO}/results/drift_summary.json"))
    print("drift decision:", s["decision"], "| drifted:", s["drifted_columns"])
    return "retrain" if s["decision"].startswith("RETRAIN") else "no_drift"


with DAG(
    dag_id="telco_churn_weekly_drift_check",
    start_date=datetime(2026, 9, 1),
    schedule="0 6 * * 1",          # Mondays 06:00
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["mlops", "drift"],
) as dag:
    drift = BashOperator(task_id="evidently_drift_check", bash_command=f"cd {REPO} && uv run python -m churn.drift")
    branch = BranchPythonOperator(task_id="drift_significant", python_callable=decide)
    retrain = BashOperator(task_id="retrain", bash_command=f"cd {REPO} && uv run python -m churn.retrain")
    no_drift = EmptyOperator(task_id="no_drift")
    drift >> branch >> [retrain, no_drift]
