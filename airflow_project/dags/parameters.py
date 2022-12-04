import os
from datetime import timedelta, datetime
from airflow.models import Variable
from airflow.utils.email import send_email_smtp


LOCAL_DATA_DIR = Variable.get('local_data_dir')
START_DATE = datetime(2022, 12, 3)


def check_file(file_name):
    return os.path.exists(file_name)


def failure_callback(context):
    dag_run = context.get('dag_run')
    subject = f'DAG {dag_run} has failed'
    send_email_smtp(to=DEFAULT_ARGS['email'], subject=subject, html_content='msg')


DEFAULT_ARGS = {
    'owner': 'romanzilotov',
    'email': ['roman.zil40@gmail.com'],
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': failure_callback
}
