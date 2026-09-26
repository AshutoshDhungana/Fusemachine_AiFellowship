"""Airflow DAG (bonus): nightly W16 harness + regression suite for the production prompt config.

Flags degradation when task_completion_rate drops > COMPLETION_DROP below the best historical
production run, or when the Evidently regression suite fails. Alerts go to the task log and,
if SLACK_WEBHOOK_URL is set, to Slack.

Setup: AIRFLOW_HOME=... ; set Airflow variable `paperpilot_repo` to the repo path; copy this file to dags/.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

REPO = Variable.get("paperpilot_repo", default_var="/opt/paperpilot")
PROD_CONFIG = Variable.get("paperpilot_prod_config", default_var="mlops/configs/v3.yaml")
COMPLETION_DROP = 0.10
TOKENS_RISE = 0.30


def check_degradation(**_):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{REPO}/mlflow.db")
    runs = mlflow.search_runs(experiment_names=["paperpilot-agent-prompts"],
                              order_by=["attributes.start_time DESC"])
    latest = runs.iloc[0]
    history = runs.iloc[1:]
    best = history["metrics.task_completion_rate"].max() if len(history) else latest["metrics.task_completion_rate"]
    base_tokens = history["metrics.avg_tokens"].median() if len(history) else latest["metrics.avg_tokens"]
    alerts = []
    if latest["metrics.task_completion_rate"] < best - COMPLETION_DROP:
        alerts.append(f"completion {latest['metrics.task_completion_rate']:.2f} < best {best:.2f} - {COMPLETION_DROP}")
    if latest["metrics.avg_tokens"] > base_tokens * (1 + TOKENS_RISE):
        alerts.append(f"avg tokens {latest['metrics.avg_tokens']:.0f} > {1 + TOKENS_RISE:.0%} of median {base_tokens:.0f}")
    if latest.get("metrics.pct_tests_passed", 1.0) < 1.0:
        alerts.append(f"regression suite pct_tests_passed={latest['metrics.pct_tests_passed']:.2f}")
    if alerts:
        msg = "PaperPilot nightly eval DEGRADED: " + "; ".join(alerts)
        if hook := os.getenv("SLACK_WEBHOOK_URL"):
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(hook, data=json.dumps({"text": msg}).encode(),
                                                          headers={"Content-Type": "application/json"}))
        raise AirflowFailException(msg)
    print("no degradation", latest[["metrics.task_completion_rate", "metrics.avg_tokens"]].to_dict())


with DAG(
    dag_id="paperpilot_nightly_regression_eval",
    start_date=datetime(2026, 9, 1),
    schedule="0 2 * * *",           # 02:00 every night
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["mlops", "llm-eval"],
) as dag:
    harness = BashOperator(
        task_id="run_harness_and_log_mlflow",
        bash_command=f"cd {REPO} && uv run python -m mlops.run_experiment --config {PROD_CONFIG}",
        execution_timeout=timedelta(hours=1),
    )
    regression = BashOperator(
        task_id="evidently_regression_suite",
        bash_command=f"cd {REPO} && uv run python -m mlops.regression_suite --config {PROD_CONFIG}",
        execution_timeout=timedelta(hours=1),
    )
    check = PythonOperator(task_id="check_degradation", python_callable=check_degradation,
                           trigger_rule="all_done")
    harness >> regression >> check
