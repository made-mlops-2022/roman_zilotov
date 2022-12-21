from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from parameters import LOCAL_DATA_DIR, DEFAULT_ARGS, START_DATE


with DAG(
    'gen_data',
    default_args=DEFAULT_ARGS,
    schedule_interval='@daily',
    start_date=START_DATE
) as dag:
    gen = DockerOperator(
        image='airflow-gen-data',
        command='--output-dir /data/raw/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-gen-data',
        do_xcom_push=False,
        auto_remove=True,
        mounts=[Mount(source=LOCAL_DATA_DIR, target='/data', type='bind')]
    )

    gen
