import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@click.command('airflow-validate')
@click.option('--input-dir', type=click.Path())
@click.option('--input-model-dir', type=click.Path())
@click.option('--output-metrics-dir', type=click.Path())
def validate(input_dir, model_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    X = pd.read_csv(os.path.join(input_dir, 'features_val.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'target_val.csv'))

    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y, y_pred)
    metrics['f1_score'] = f1_score(y, y_pred)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    validate()
