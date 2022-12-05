from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount
from parameters import LOCAL_DATA_DIR, DEFAULT_ARGS, check_file, START_DATE


with DAG(
    'train',
    default_args=DEFAULT_ARGS,
    schedule_interval='@weekly',
    start_date=START_DATE
) as dag:
    wait_for_data = PythonSensor(
        task_id='wait-for-data',
        python_callable=check_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    preprocessing_and_split = DockerOperator(
        image='airflow-preprocessing-and-split',
        command='--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-preprocessing-and-split',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    train = DockerOperator(
        image='airflow-train',
        command='--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}',
        network_mode='host',
        task_id='docker-airflow-train',
        do_xcom_push=True,
        auto_remove=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    validate = DockerOperator(
        image='airflow-validate',
        command='--input-dir /data/processed/{{ ds }} --input-model-dir /data/models/{{ ds }} --output-metrics-dir /data/metrics/{{ ds }}',
        network_mode='host',
        task_id='docker-airflow-validate',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    wait_for_data >> preprocessing_and_split >> train >> validate
