
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.operators.bash import BashOperator


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_DIR = "/opt/airflow/project"

PROJECT_PYTHON = "/home/airflow/project-venv/bin/python"
DVC_EXECUTABLE = "/home/airflow/project-venv/bin/dvc"

INCOMING_FILE = "sales_data_drift.csv"

INCOMING_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "incoming",
    INCOMING_FILE
)


# ==========================================
# 2. SENSOR FUNCTION
# ==========================================

def incoming_file_available():
    return os.path.isfile(INCOMING_PATH)


# ==========================================
# 3. DEFAULT ARGUMENTS
# ==========================================

default_args = {
    "owner": "mlops-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5)
}


# ==========================================
# 4. DEFINE DAG
# ==========================================

with DAG(
    dag_id="mlops_retraining",
    default_args=default_args,
    description="Data arrival and automated DVC retraining",
    schedule=None,
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["mlops", "dvc", "data-arrival"]
) as dag:

    # ======================================
    # TASK 1: WAIT FOR INCOMING DATA
    # ======================================

    wait_for_data = PythonSensor(
        task_id="wait_for_sales_data",
        python_callable=incoming_file_available,
        poke_interval=10,
        timeout=900,
        mode="reschedule"
    )

    # ======================================
    # TASK 2: INGEST DATA AND RUN DVC
    # ======================================

    ingest_and_train = BashOperator(
        task_id="ingest_and_run_dvc",

        bash_command=f"""
            set -e

            cd {PROJECT_DIR}

            git config --global --add safe.directory {PROJECT_DIR}

            echo "Copying incoming dataset..."

            cp "{INCOMING_PATH}" data/raw/sales.csv

            echo "Updating DVC dataset version..."

            {DVC_EXECUTABLE} add data/raw/sales.csv

            echo "Executing DVC pipeline..."

            {DVC_EXECUTABLE} repro

            echo "Pushing updated DVC artifacts..."

            {DVC_EXECUTABLE} push

            mkdir -p data/archive

            mv "{INCOMING_PATH}" "data/archive/{INCOMING_FILE}"

            echo "Data ingestion and training completed."
        """,

        env={
            "PATH": "/home/airflow/project-venv/bin:/usr/local/bin:/usr/bin:/bin"
        },
        append_env=True
    )

    # ======================================
    # TASK DEPENDENCY
    # ======================================

    wait_for_data >> ingest_and_train
